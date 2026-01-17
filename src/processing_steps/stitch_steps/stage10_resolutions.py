#!/usr/bin/env python3
"""
Stage 10: LLM Resolution for Remaining UNKNOWN Words
====================================================

This stage handles two types of unresolved speaker assignments:
1. NEEDS_EMBEDDING: Uses speaker embeddings to find closest speaker centroid
2. NEEDS_LLM: Uses LLM analysis for complex cases

Key Responsibilities:
- Process all words still marked as UNKNOWN after previous stages
- Use LLM with conversation context for speaker assignment
- Handle edge cases and complex conversational patterns
- Apply final resolution for any remaining unassigned words

Input:
- WordTable from Stage 9 with some words still marked as NEEDS_EMBEDDING OR NEEDS_LLM
- Conversation context surrounding uncertain words
- Available speaker options from previous assignments

Output:
- WordTable with final speaker assignments for NEEDS_EMBEDDING OR NEEDS_LLM words
- LLM resolution statistics including confidence scores
- Assignment reasoning and context analysis

Key Components:
- LLMSpeakerResolver: Main class for LLM-based speaker resolution
- Context window analysis: Uses surrounding conversation for better decisions
- Multi-model support: Primary and fallback LLM models
- Temperature-based retries: Increases creativity for difficult cases

Methods:
- llm_resolution_stage(): Main entry point called by stitch pipeline
- LLMSpeakerResolver.resolve_unknown_words(): Core LLM processing logic
- build_conversation_context(): Creates context windows for LLM analysis

Performance:
- Moderate computational cost (10-15% of pipeline time)
- Uses local LLM models via Ollama
- Context-aware processing for better accuracy
- Efficient batch processing where possible
"""

import logging
import time
import json
import re
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import setup_worker_logger
from src.utils.config import get_temperature_config
from src.utils.llm_client import LLMClient, create_stitch_client
from src.processing_steps.stitch_steps.stage3_tables import WordTable
from src.processing_steps.stitch_steps.util_stitch import update_word_assignment, summarize_speaker_assignments
from src.storage.s3_utils import S3Storage
# Model server imports removed - using direct model loading only

try:
    import torch
    from pyannote.audio import Inference
    import librosa
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

logger = setup_worker_logger('stitch')
logger.setLevel(logging.INFO)


class EmbeddingResolver:
    """
    Resolves NEEDS_EMBEDDING segments by extracting audio embeddings and comparing with speaker centroids.
    """
    
    def __init__(self, speaker_centroids: Dict[str, np.ndarray], 
                 similarity_threshold: float = 0.5,
                 audio_path: Optional[str] = None,
                 s3_storage: Optional[S3Storage] = None):
        """
        Initialize the embedding resolver.
        
        Args:
            speaker_centroids: Dictionary mapping speaker IDs to centroid embeddings
            similarity_threshold: Minimum similarity for speaker assignment (default 0.5)
            audio_path: Path to the audio file (local or S3)
            s3_storage: S3 storage instance for accessing audio
        """
        self.speaker_centroids = speaker_centroids
        self.similarity_threshold = similarity_threshold
        self.audio_path = audio_path
        self.s3_storage = s3_storage
        self.embedding_model = None
        self.use_local_model = True  # Always use local model (model server deprecated)
        self.audio_data = None
        self.sample_rate = None
        self._temp_audio_file = None
        
        # Initialize local model if available
        self._initialize_local_model()
        
        # Validate we have centroids
        if not speaker_centroids:
            logger.warning("No speaker centroids provided for embedding resolution")
    
    def _initialize_local_model(self):
        """Initialize local pyannote model if available."""
        # Try local pyannote model
        if PYANNOTE_AVAILABLE:
            try:
                logger.info("Loading local pyannote embedding model...")
                self.embedding_model = Inference("pyannote/embedding", window="whole")
                self.use_local_model = True
                logger.info("Successfully loaded local pyannote embedding model")
            except Exception as local_error:
                logger.warning(f"Failed to load local pyannote model during init: {local_error}")
                # Don't fail here - we'll try again when actually needed
                self.use_local_model = False
                self.embedding_model = None
        else:
            logger.warning("Pyannote not installed - embedding resolution will not be available")
            # Don't fail initialization - embeddings might not be needed
    
    async def load_audio(self, content_id: str) -> bool:
        """
        Load the audio file if not already loaded.

        Args:
            content_id: Content ID for logging

        Returns:
            True if audio loaded successfully
        """
        if self.audio_data is not None:
            return True

        try:
            local_path = None
            if self.audio_path:
                # Check if it's a local file path
                if os.path.exists(self.audio_path):
                    # Load from local path directly
                    self.audio_data, self.sample_rate = librosa.load(self.audio_path, sr=16000, mono=True)
                    logger.info(f"[{content_id}] Loaded audio file from local path: {len(self.audio_data)/self.sample_rate:.1f}s at {self.sample_rate}Hz")
                    return True
                elif self.audio_path.startswith('s3://') and self.s3_storage:
                    # Download from S3
                    local_path = f"/tmp/{content_id}_audio.wav"
                    # Use flexible download that handles multiple formats
                    if not self.s3_storage.download_audio_flexible(content_id, local_path):
                        logger.error(f"[{content_id}] Failed to download audio from S3")
                        return False
                    self.audio_data, self.sample_rate = librosa.load(local_path, sr=16000, mono=True)
                    # Store the temp file path for later cleanup
                    self._temp_audio_file = local_path
                    logger.info(f"[{content_id}] Loaded audio from S3: {len(self.audio_data)/self.sample_rate:.1f}s at {self.sample_rate}Hz")
                    return True
                else:
                    logger.warning(f"[{content_id}] Audio path not found: {self.audio_path}, will try S3")

            # Try to get audio from S3 using flexible download
            if self.s3_storage:
                local_path = f"/tmp/{content_id}_audio.wav"
                # Use flexible download that handles opus/mp3/wav formats
                if not self.s3_storage.download_audio_flexible(content_id, local_path):
                    logger.error(f"[{content_id}] Failed to download audio from S3")
                    return False
                self.audio_data, self.sample_rate = librosa.load(local_path, sr=16000, mono=True)
                # Store the temp file path for later cleanup
                self._temp_audio_file = local_path
                logger.info(f"[{content_id}] Loaded audio from S3: {len(self.audio_data)/self.sample_rate:.1f}s")
                return True
            else:
                logger.error(f"[{content_id}] No audio path provided and no S3 storage available")
                return False

        except Exception as e:
            logger.error(f"[{content_id}] Failed to load audio: {str(e)}")
            # Clean up temp file if it was created but loading failed
            if local_path and os.path.exists(local_path):
                try:
                    os.unlink(local_path)
                except:
                    pass
            return False
    
    async def get_segment_embedding(self, segment: Dict[str, Any], content_id: str) -> Optional[np.ndarray]:
        """
        Get audio embedding for a segment.
        
        Args:
            segment: Segment information including start/end times
            content_id: Content ID for logging
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            # Ensure audio is loaded
            if not await self.load_audio(content_id):
                return None
            
            # Extract segment audio
            start_time = segment.get('start_time', 0)
            end_time = segment.get('end_time', 0)
            
            start_sample = int(start_time * self.sample_rate)
            end_sample = int(end_time * self.sample_rate)
            
            # Add small padding to avoid edge effects
            padding_samples = int(0.1 * self.sample_rate)  # 100ms padding
            start_sample = max(0, start_sample - padding_samples)
            end_sample = min(len(self.audio_data), end_sample + padding_samples)
            
            segment_audio = self.audio_data[start_sample:end_sample]
            
            if len(segment_audio) < self.sample_rate * 0.1:  # Less than 100ms
                logger.warning(f"[{content_id}] Segment too short for embedding: {len(segment_audio)/self.sample_rate:.3f}s")
                return None
            
            if self.use_local_model:
                # Use local pyannote model
                logger.debug(f"[{content_id}] Using local pyannote embedding model")
                if self.embedding_model is None:
                    self.embedding_model = Inference("pyannote/embedding", window="whole")
                
                # Create a temporary dict for pyannote
                waveform = torch.from_numpy(segment_audio).unsqueeze(0)
                sample_dict = {"waveform": waveform, "sample_rate": self.sample_rate}
                
                # Get embedding
                embedding = self.embedding_model(sample_dict)
                
                # Handle different return types
                if hasattr(embedding, 'detach'):
                    # It's a tensor
                    embedding_array = embedding.detach().cpu().numpy()
                elif isinstance(embedding, np.ndarray):
                    # It's already a numpy array
                    embedding_array = embedding
                else:
                    # Convert to numpy array
                    embedding_array = np.array(embedding)
                
                # Normalize
                embedding_array = embedding_array / np.linalg.norm(embedding_array)
                return embedding_array
            else:
                # This should not happen since we always use local model now
                logger.error(f"[{content_id}] Unexpected state: use_local_model is False but we should always use local model")
                return None
            
        except Exception as e:
            logger.error(f"[{content_id}] Failed to get audio embedding: {str(e)}")
            
            # Try fallback to local model if server fails
            if not self.use_local_model and PYANNOTE_AVAILABLE:
                try:
                    logger.warning(f"[{content_id}] Attempting fallback to local pyannote model")
                    # Try to load the model if not already loaded
                    if self.embedding_model is None:
                        try:
                            self.embedding_model = Inference("pyannote/embedding", window="whole")
                            logger.info(f"[{content_id}] Successfully loaded local pyannote embedding model for fallback")
                        except Exception as load_error:
                            logger.error(f"[{content_id}] Failed to load local pyannote model: {str(load_error)}")
                            return None
                    
                    self.use_local_model = True
                    # Recursive call to try again with local model
                    return await self.get_segment_embedding(segment, content_id)
                except Exception as local_e:
                    logger.error(f"[{content_id}] Local model fallback also failed: {str(local_e)}")
            
            return None
    
    async def get_audio_embedding(self, segment: Dict[str, Any], content_id: str) -> Optional[np.ndarray]:
        """
        Alias for get_segment_embedding to maintain compatibility.
        
        Args:
            segment: Segment information including start/end times
            content_id: Content ID for logging
            
        Returns:
            Embedding vector or None if failed
        """
        return await self.get_segment_embedding(segment, content_id)
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        if hasattr(self, '_temp_audio_file') and self._temp_audio_file and os.path.exists(self._temp_audio_file):
            try:
                os.unlink(self._temp_audio_file)
                logger.debug(f"Cleaned up temporary audio file: {self._temp_audio_file}")
                self._temp_audio_file = None
            except Exception as e:
                logger.warning(f"Failed to clean up temporary audio file {self._temp_audio_file}: {e}")
        
        # Clear audio data from memory
        self.audio_data = None
        self.sample_rate = None
    
    def find_closest_speaker(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find the closest speaker centroid to the given embedding.
        
        Args:
            embedding: Segment embedding vector
            
        Returns:
            Tuple of (speaker_id, similarity_score) or (None, 0.0) if no match
        """
        if not self.speaker_centroids:
            return None, 0.0
        
        # Normalize the embedding
        embedding_norm = embedding / np.linalg.norm(embedding)
        
        # Calculate similarities for all speakers
        similarities = []
        for speaker_id, centroid in self.speaker_centroids.items():
            # Skip non-speaker IDs
            if not speaker_id.startswith('SPEAKER_'):
                continue
                
            # Normalize centroid
            centroid_norm = centroid / np.linalg.norm(centroid)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding_norm, centroid_norm)
            similarities.append((speaker_id, similarity))
        
        if not similarities:
            return None, 0.0
            
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        best_speaker, best_similarity = similarities[0]
        
        # Check absolute threshold (>0.4)
        if best_similarity <= 0.4:
            return None, best_similarity
        
        # Check margin requirement (0.15 above next best)
        if len(similarities) > 1:
            second_best_similarity = similarities[1][1]
            margin = best_similarity - second_best_similarity
            if margin < 0.15:
                return None, best_similarity
        
        return best_speaker, best_similarity
    
    async def resolve_segment(self, segment: Dict[str, Any], content_id: str) -> Dict[str, Any]:
        """
        Resolve a NEEDS_EMBEDDING segment using audio embeddings.
        
        Args:
            segment: Segment information including start/end times and word indices
            content_id: Content ID
            
        Returns:
            Resolution result with speaker assignment or failure info
        """
        try:
            # Check segment has timing information
            if 'start_time' not in segment or 'end_time' not in segment:
                return {
                    'status': 'error',
                    'reason': 'No timing information for segment',
                    'speaker': None,
                    'similarity': 0.0
                }
            
            duration = segment['end_time'] - segment['start_time']
            if duration < 0.1:  # Less than 100ms
                return {
                    'status': 'error',
                    'reason': f'Segment too short: {duration:.3f}s',
                    'speaker': None,
                    'similarity': 0.0
                }
            
            # Get audio embedding for the segment
            embedding = await self.get_segment_embedding(segment, content_id)
            if embedding is None:
                return {
                    'status': 'error',
                    'reason': 'Failed to get audio embedding',
                    'speaker': None,
                    'similarity': 0.0
                }
            
            # Find closest speaker
            speaker, similarity = self.find_closest_speaker(embedding)
            
            if speaker:
                logger.info(f"[{content_id}] Matched audio segment ({segment['start_time']:.2f}-{segment['end_time']:.2f}s) "
                          f"to {speaker} with similarity {similarity:.3f}")
                return {
                    'status': 'success',
                    'speaker': speaker,
                    'similarity': float(similarity),
                    'method': 'audio_embedding_similarity',
                    'duration': duration
                }
            else:
                logger.info(f"[{content_id}] No speaker match above threshold for audio segment "
                          f"({segment['start_time']:.2f}-{segment['end_time']:.2f}s) - best: {similarity:.3f}")
                return {
                    'status': 'below_threshold',
                    'speaker': None,
                    'similarity': float(similarity),
                    'reason': f'Best similarity {similarity:.3f} below threshold {self.similarity_threshold}'
                }
                
        except Exception as e:
            logger.error(f"[{content_id}] Error in audio embedding resolution: {str(e)}")
            return {
                'status': 'error',
                'reason': str(e),
                'speaker': None,
                'similarity': 0.0
            }


class Stage10LLMResolver:
    """
    LLM resolver for Stage 10 - handling NEEDS_LLM segments and
    embedding segments that couldn't be resolved with high confidence.

    Uses the unified LLM client for all requests through the balancer.
    """

    def __init__(self, model: str = None, llm_server_url: str = None):
        # Use tier_2 for stitch tasks with priority 1 (real-time)
        self.model = model or "tier_2"

        # Get temperature configuration
        self.temp_config = get_temperature_config('speaker_assignment')

        # Create unified LLM client for stitch tasks
        self._llm_client: Optional[LLMClient] = None

        logger.info(f"Stage10LLMResolver initialized with tier: {self.model}")

    async def _get_llm_client(self) -> LLMClient:
        """Get or create the LLM client."""
        if self._llm_client is None:
            self._llm_client = create_stitch_client(tier=self.model)
        return self._llm_client

    async def close(self):
        """Close the LLM client session."""
        if self._llm_client is not None:
            await self._llm_client.close()
            self._llm_client = None
    
    async def resolve_unknown_segment(self, context: Dict[str, Any], test_mode: bool = False) -> Dict[str, Any]:
        """
        Resolve a segment of UNKNOWN words using comprehensive LLM analysis.
        
        Args:
            context: Comprehensive context about the segment
            test_mode: Whether to enable detailed logging
            
        Returns:
            Dictionary with resolution results
        """
        try:
            # Create comprehensive prompt
            prompt = self._create_resolution_prompt(context)
            
            if test_mode:
                logger.info("=" * 80)
                logger.info("STAGE 8 LLM RESOLUTION PROMPT")
                logger.info("=" * 80)
                logger.info(prompt)
                logger.info("=" * 80)
            
            # Get LLM response
            response = await self._get_llm_response(prompt, test_mode=test_mode)
            
            if test_mode:
                logger.info("STAGE 8 LLM RESPONSE:")
                logger.info(response)
                logger.info("=" * 80)
            
            if not response:
                return {
                    'status': 'error',
                    'reason': 'No response from LLM',
                    'assignments': {}
                }
            
            # Parse response
            assignments = self._parse_resolution_response(response, context)
            
            return {
                'status': 'success',
                'assignments': assignments,
                'llm_response': response
            }
            
        except Exception as e:
            logger.error(f"Error in Stage 8 LLM resolution: {str(e)}")
            return {
                'status': 'error',
                'reason': str(e),
                'assignments': {}
            }
    
    def _create_resolution_prompt(self, context: Dict[str, Any]) -> str:
        """Create comprehensive prompt for Stage 8 resolution."""
        # Extract key information
        unknown_words = context['unknown_segment']['words']
        before_context = context.get('before_context', [])
        after_context = context.get('after_context', [])
        available_speakers = context.get('available_speakers', [])
        diarization_info = context.get('diarization_info', [])
        
        # Build context strings
        before_text = self._format_context_words(before_context[-15:])  # More context for Stage 8
        after_text = self._format_context_words(after_context[:15])
        unknown_text = ' '.join([w['word'] for w in unknown_words])
        
        # Diarization analysis - include overlaps and nearby segments
        diar_section = ""
        if diarization_info:
            diar_section = "\nDIARIZATION EVIDENCE:\n"
            
            for diar in diarization_info:
                if diar.get('context_type') == 'overlap':
                    diar_section += f"- {diar['speaker']}: {diar['start']:.2f}s-{diar['end']:.2f}s "
                    diar_section += f"(overlaps {diar['overlap_percentage']:.0%})\n"
                else:
                    # Nearby segment
                    diar_section += f"- {diar['speaker']}: {diar['start']:.2f}s-{diar['end']:.2f}s"
                    # Add distance info
                    if diar['end'] <= unknown_words[0]['start']:
                        distance = unknown_words[0]['start'] - diar['end']
                        diar_section += f" ({distance:.1f}s before)"
                    elif diar['start'] >= unknown_words[-1]['end']:
                        distance = diar['start'] - unknown_words[-1]['end']
                        diar_section += f" ({distance:.1f}s after)"
                    diar_section += "\n"
        
        # Word timing details
        timing_section = "\nWORD TIMINGS:\n"
        for i, word in enumerate(unknown_words):
            timing_section += f"- Word {i+1} '{word['word']}': {word['start']:.2f}s-{word['end']:.2f}s\n"
                
        prompt = f"""TASK: Assign speakers to the remaining UNKNOWN words.

CONVERSATION CONTEXT:
{before_text}
[UNKNOWN] {unknown_text}
{after_text}

{diar_section}
{timing_section}

RESOLUTION GUIDELINES:
1. Apply careful reasoning to these challenging cases
2. Consider conversational patterns and natural flow as primary evidence
3. Use diarization timing as evidence when available
4. Look for speaker transitions and conversation continuity

Reply with a JSON object in this exact format:
{{
  "assignments": ["""

        # Add each word
        for i, word in enumerate(unknown_words):
            prompt += f'\n    {{"word_index": {i}, "text": "{word["word"]}", "speaker": "SPEAKER_XX"}},'
        
        # Remove last comma and close
        if unknown_words:
            prompt = prompt.rstrip(',')
        
        prompt += """
  ]
}"""
        
        return prompt
    
    def _format_context_words(self, words: List[Dict]) -> str:
        """Format context words for display."""
        if not words:
            return ""
        
        # Group by speaker for cleaner display
        result = ""
        current_speaker = None
        current_text = []
        
        for word in words:
            speaker = word.get('speaker', 'UNKNOWN')
            if speaker != current_speaker:
                if current_text:
                    speaker_label = current_speaker if current_speaker not in ['UNKNOWN', 'MULTI_SPEAKER'] else '?'
                    result += f"[{speaker_label}] {' '.join(current_text)} "
                current_speaker = speaker
                current_text = [word['word']]
            else:
                current_text.append(word['word'])
        
        # Add last group
        if current_text:
            speaker_label = current_speaker if current_speaker not in ['UNKNOWN', 'MULTI_SPEAKER'] else '?'
            result += f"[{speaker_label}] {' '.join(current_text)}"
        
        return result.strip()
    
    async def _get_llm_response(self, prompt: str, max_retries: int = 3, test_mode: bool = False) -> Optional[str]:
        """Get response from LLM with retry logic using unified client."""
        llm_client = await self._get_llm_client()

        for attempt in range(max_retries):
            try:
                # Calculate temperature with increment on retry
                initial_temp = self.temp_config.get('initial', 0.1)
                increment = self.temp_config.get('increment', 0.2)
                max_temp = self.temp_config.get('max', 0.5)
                temperature = min(initial_temp + (attempt * increment), max_temp)

                if test_mode:
                    logger.info(f"Stage 10 LLM Call - Attempt {attempt + 1}/{max_retries} (temperature: {temperature:.1f})")

                # Prepare messages
                messages = [
                    {"role": "system", "content": "You are an expert transcript editor performing final speaker resolution. Provide detailed reasoning and accurate JSON responses. Do not use thinking tags or internal monologue."},
                    {"role": "user", "content": prompt}
                ]

                # Use unified client
                response = await llm_client.call(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=200,
                    retries=1,  # We handle retries at this level with temp increase
                )

                if response:
                    if test_mode:
                        logger.info(f"Stage 10 LLM Call - Response: success=True, length={len(response)}")
                        logger.info(f"Stage 10 LLM Call - Preview: {response[:200]}...")
                    return response.strip()
                else:
                    logger.warning("LLM returned empty response")

            except Exception as e:
                logger.error(f"Stage 10 LLM error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return None

        return None
    
    def _parse_resolution_response(self, response: str, context: Dict[str, Any]) -> Dict[int, str]:
        """Parse Stage 8 LLM response and extract assignments."""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON found in Stage 8 response")
                return {}
            
            json_str = response[json_start:json_end]
            
            # Clean up common issues
            json_str = re.sub(r'\n\s*', ' ', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            
            parsed = json.loads(json_str)
            
            
            # Extract assignments
            assignments = {}
            unknown_words = context['unknown_segment']['words']
            
            for assignment in parsed.get('assignments', []):
                word_index = assignment.get('word_index')
                speaker = assignment.get('speaker')
                
                if word_index is not None and speaker and word_index < len(unknown_words):
                    word_info = unknown_words[word_index]
                    word_idx = word_info['word_idx']
                    assignments[word_idx] = speaker
                    
                    logger.debug(f"Stage 8: Word {word_index} '{word_info['word']}' → {speaker}")
            
            return assignments
            
        except Exception as e:
            logger.error(f"Failed to parse Stage 8 response: {str(e)}")
            logger.debug(f"Raw response: {response[:500]}...")
            return {}


def identify_unresolved_segments(word_table: WordTable, target_labels: List[str]) -> List[Dict[str, Any]]:
    """
    Identify all segments with specified labels (NEEDS_EMBEDDING, NEEDS_LLM, UNKNOWN).
    
    Args:
        word_table: WordTable instance
        target_labels: List of labels to identify
        
    Returns:
        List of segments with context information
    """
    segments = []
    current_segment = []
    current_label = None
    segment_index = 0
    
    df = word_table.df
    
    for idx, word in df.iterrows():
        speaker = word['speaker_current']
        
        if speaker in target_labels:
            if current_segment and speaker != current_label:
                # End current segment if label changed
                segments.append({
                    'segment_index': segment_index,
                    'label': current_label,
                    'words': current_segment,
                    'start_time': current_segment[0]['start'],
                    'end_time': current_segment[-1]['end'],
                    'text': ' '.join([w['word'] for w in current_segment]),
                    'word_count': len(current_segment)
                })
                segment_index += 1
                current_segment = []
            
            current_label = speaker
            current_segment.append({
                'word_idx': idx,
                'word': word['text'],
                'start': word['start'],
                'end': word['end'],
                'segment_index': word['segment_index'],
                'original_label': speaker
            })
        else:
            # End of target segment
            if current_segment:
                segments.append({
                    'segment_index': segment_index,
                    'label': current_label,
                    'words': current_segment,
                    'start_time': current_segment[0]['start'],
                    'end_time': current_segment[-1]['end'],
                    'text': ' '.join([w['word'] for w in current_segment]),
                    'word_count': len(current_segment)
                })
                segment_index += 1
                current_segment = []
                current_label = None
    
    # Handle final segment
    if current_segment:
        segments.append({
            'segment_index': segment_index,
            'label': current_label,
            'words': current_segment,
            'start_time': current_segment[0]['start'],
            'end_time': current_segment[-1]['end'],
            'text': ' '.join([w['word'] for w in current_segment]),
            'word_count': len(current_segment)
        })
    
    return segments


def try_simple_resolution(segment: Dict[str, Any], word_table: WordTable,
                         diarization_data: Optional[Dict] = None) -> Optional[str]:
    """
    Try to resolve segment using simple heuristics before invoking audio embedding analysis.

    Uses conservative heuristics to avoid false positives:
    - Case 1: Diarization overlap - ANY segment length (safe, backed by audio analysis)
    - Case 2: Sandwiched in transcript - ONLY single words (multi-word could be interjections)
    - Case 3: Closest diarization - ONLY single words (multi-word needs audio verification)

    Returns:
        Speaker ID if simple resolution possible with high confidence, None otherwise
    """
    # Get diarization segments that overlap with this segment
    if not diarization_data or 'segments' not in diarization_data:
        return None
        
    overlapping_speakers = []
    for diar_seg in diarization_data['segments']:
        # Check overlap
        if (diar_seg['start'] < segment['end_time'] and 
            diar_seg['end'] > segment['start_time']):
            overlapping_speakers.append(diar_seg['speaker'])
    
    # Case 1: All overlapping diarization segments are from the same speaker
    # This is safe because diarization audio analysis confirms the speaker
    if overlapping_speakers and len(set(overlapping_speakers)) == 1:
        return overlapping_speakers[0]

    # Case 2: Check if segment is between the same speaker
    # ONLY apply this for single words - multi-word segments could be interjections
    if len(segment['words']) == 1:
        # Get the first and last word indices
        first_word = segment['words'][0]
        last_word = segment['words'][-1]

        # Find speakers immediately before and after
        speaker_before = None
        speaker_after = None

        # Look at surrounding words in the word table
        df = word_table.df
        first_idx = first_word['word_idx']
        last_idx = last_word['word_idx']

        # Check word before first word
        if first_idx > 0 and first_idx - 1 in df.index:
            prev_word = df.loc[first_idx - 1]
            if prev_word['speaker_current'].startswith('SPEAKER_'):
                speaker_before = prev_word['speaker_current']

        # Check word after last word
        if last_idx + 1 in df.index:
            next_word = df.loc[last_idx + 1]
            if next_word['speaker_current'].startswith('SPEAKER_'):
                speaker_after = next_word['speaker_current']

        # If both are the same speaker, assign to that speaker
        if speaker_before and speaker_before == speaker_after:
            return speaker_before

    # Case 3: Check closest diarization segments before and after
    # ONLY apply this for single words - multi-word segments need audio analysis
    if len(segment['words']) == 1:
        segment_start = segment['start_time']
        segment_end = segment['end_time']

        closest_before_speaker = None
        closest_before_distance = float('inf')
        closest_after_speaker = None
        closest_after_distance = float('inf')

        for diar_seg in diarization_data['segments']:
            # Check if this is before the segment
            if diar_seg['end'] <= segment_start:
                distance = segment_start - diar_seg['end']
                if distance < closest_before_distance:
                    closest_before_distance = distance
                    closest_before_speaker = diar_seg['speaker']

            # Check if this is after the segment
            if diar_seg['start'] >= segment_end:
                distance = diar_seg['start'] - segment_end
                if distance < closest_after_distance:
                    closest_after_distance = distance
                    closest_after_speaker = diar_seg['speaker']

        # If the closest speakers before and after are the same, assign to that speaker
        if closest_before_speaker and closest_before_speaker == closest_after_speaker:
            return closest_before_speaker
    
    return None


def gather_segment_context(segment: Dict[str, Any], word_table: WordTable, 
                          diarization_data: Optional[Dict] = None,
                          context_window: int = 20) -> Dict[str, Any]:
    """
    Gather comprehensive context for an unknown segment.
    
    For NEEDS_LLM segments, this includes:
    - All diarization segments that overlap with the segment
    - Adjacent diarization segments (immediately before/after overlapping ones)
    - Extended neighboring segments within threshold
    
    Args:
        segment: The unknown segment to analyze
        word_table: WordTable instance
        diarization_data: Optional diarization data
        context_window: Number of words to include before/after
        
    Returns:
        Dictionary with comprehensive context
    """
    df = word_table.df
    first_idx = segment['words'][0]['word_idx']
    last_idx = segment['words'][-1]['word_idx']
    
    # Get context before
    before_context = []
    for i in range(max(0, first_idx - context_window), first_idx):
        if i in df.index:
            word = df.loc[i]
            before_context.append({
                'word': word['text'],
                'speaker': word['speaker_current'],
                'start': word['start'],
                'end': word['end']
            })
    
    # Get context after
    after_context = []
    for i in range(last_idx + 1, min(len(df), last_idx + context_window + 1)):
        if i in df.index:
            word = df.loc[i]
            after_context.append({
                'word': word['text'],
                'speaker': word['speaker_current'],
                'start': word['start'],
                'end': word['end']
            })
    
    # Get available speakers
    speakers = set(df['speaker_current'].unique())
    speakers.discard('UNKNOWN')
    speakers.discard('MULTI_SPEAKER')
    speakers.discard('NEEDS_EMBEDDING')
    speakers.discard('NEEDS_LLM')
    available_speakers = sorted([s for s in speakers if s.startswith('SPEAKER_')])
    
    # Get diarization info from segments near the unknown words
    diarization_info = []
    if diarization_data and 'segments' in diarization_data:
        diar_segments = diarization_data['segments']
        
        # Expand segment boundaries by 4 seconds on each side
        expanded_start = segment['start_time'] - 4.0
        expanded_end = segment['end_time'] + 4.0
        
        for i, diar_seg in enumerate(diar_segments):
            # Include any diarization segment that touches the expanded range
            if diar_seg['start'] < expanded_end and diar_seg['end'] > expanded_start:
                # Check if this actually overlaps with the original (non-expanded) segment
                overlaps = (diar_seg['start'] < segment['end_time'] and diar_seg['end'] > segment['start_time'])
                
                if overlaps:
                    overlap_start = max(diar_seg['start'], segment['start_time'])
                    overlap_end = min(diar_seg['end'], segment['end_time'])
                    overlap_duration = overlap_end - overlap_start
                    segment_duration = segment['end_time'] - segment['start_time']
                    overlap_percentage = overlap_duration / segment_duration if segment_duration > 0 else 0
                    context_type = 'overlap'
                else:
                    # This is nearby but not overlapping
                    overlap_duration = 0
                    overlap_percentage = 0
                    context_type = 'nearby'
                    
                    # Recalculate distances for non-overlapping segments
                    distance_before = segment['start_time'] - diar_seg['end']
                    distance_after = diar_seg['start'] - segment['end_time']
                
                # Calculate distance from unknown segment
                if overlaps:
                    distance = 0
                else:
                    distance = min(abs(distance_before), abs(distance_after))
                
                diarization_info.append({
                    'speaker': diar_seg['speaker'],
                    'start': diar_seg['start'],
                    'end': diar_seg['end'],
                    'overlap_duration': overlap_duration,
                    'overlap_percentage': overlap_percentage,
                    'overlaps_unknown': overlaps,
                    'context_type': context_type,
                    'distance': distance
                })
        
        # Sort diarization info by start time
        diarization_info.sort(key=lambda x: x['start'])
    
    return {
        'unknown_segment': segment,
        'before_context': before_context,
        'after_context': after_context,
        'available_speakers': available_speakers,
        'diarization_info': diarization_info
    }


async def stage10_resolutions(content_id: str,
                             word_table: WordTable,
                             speaker_centroids: Optional[Dict[str, np.ndarray]] = None,
                             diarization_data: Optional[Dict] = None,
                             s3_storage: Optional[S3Storage] = None,
                             test_mode: bool = False,
                             output_dir: Optional[Path] = None,
                             audio_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Execute Stage 10: Final resolutions for NEEDS_EMBEDDING and NEEDS_LLM segments.

    This stage first attempts embedding-based resolution for NEEDS_EMBEDDING segments,
    then uses LLM for NEEDS_LLM segments and any embedding segments below threshold.

    Args:
        content_id: Content identifier
        word_table: Word table with current assignments
        speaker_centroids: Speaker centroid embeddings from Stage 6/7
        diarization_data: Optional diarization data
        s3_storage: S3 storage instance (for accessing audio/centroids if needed)
        test_mode: Whether to run in test mode with detailed logging
        output_dir: Optional output directory for logs
        audio_path: Optional path to local audio file (downloaded in Stage 1)

    Returns:
        Dictionary with stage results
    """
    start_time = time.time()
    
    logger.info(f"[{content_id}] Starting Stage 10: Final Resolutions for NEEDS_EMBEDDING and NEEDS_LLM")
    
    result = {
        'status': 'pending',
        'content_id': content_id,
        'stage': 'stage10_resolutions',
        'data': {
            'word_table': None,
            'resolution_results': [],
            'output_files': []
        },
        'stats': {},
        'error': None
    }
    
    try:
        # Validate inputs
        if word_table is None or word_table.df is None:
            raise ValueError("No word table available")
        
        # First, convert any remaining category labels to UNKNOWN
        category_labels = {'BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI'}
        category_mask = word_table.df['speaker_current'].isin(category_labels)
        category_count = category_mask.sum()
        
        if category_count > 0:
            logger.info(f"[{content_id}] Converting {category_count} remaining category labels to NEEDS_LLM")
            word_table.df.loc[category_mask, 'speaker_current'] = 'NEEDS_LLM'
            # Add a note in the assignment history
            for idx in word_table.df[category_mask].index:
                history = word_table.df.at[idx, 'assignment_history']
                if not isinstance(history, list):
                    history = []
                history.append({
                    'stage': 'stage10_category_cleanup',
                    'timestamp': time.time(),
                    'speaker': 'NEEDS_LLM',
                    'method': 'category_to_needs_llm',
                    'confidence': 0.0,
                    'reason': 'Category label not resolved by previous stages'
                })
                word_table.df.at[idx, 'assignment_history'] = history
        
        # Count initial unresolved words BEFORE any conversion
        target_labels = ['NEEDS_EMBEDDING', 'NEEDS_LLM', 'UNKNOWN']
        initial_embedding_count = len(word_table.df[word_table.df['speaker_current'] == 'NEEDS_EMBEDDING'])
        initial_llm_count = len(word_table.df[word_table.df['speaker_current'] == 'NEEDS_LLM'])
        initial_unknown_count = len(word_table.df[word_table.df['speaker_current'] == 'UNKNOWN'])
        initial_total = initial_embedding_count + initial_llm_count + initial_unknown_count
        
        logger.info(f"[{content_id}] Initial unresolved words: {initial_total} "
                   f"(NEEDS_EMBEDDING: {initial_embedding_count}, NEEDS_LLM: {initial_llm_count}, UNKNOWN: {initial_unknown_count})")
        
        # Log speaker centroids status
        if speaker_centroids is not None:
            logger.info(f"[{content_id}] Speaker centroids available: {len(speaker_centroids)} speakers")
        else:
            logger.warning(f"[{content_id}] No speaker centroids provided to stage10")
        
        if initial_total == 0:
            logger.info(f"[{content_id}] No unresolved words - Stage 10 complete")
            result['status'] = 'success'
            result['data']['word_table'] = word_table
            result['stats'] = {
                'duration': time.time() - start_time,
                'initial_total': 0,
                'embedding_resolved': 0,
                'llm_resolved': 0,
                'final_unresolved': 0
            }
            return result
        
        # Create output directory and log file if specified
        resolution_log_file = None
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / f"stage10_resolution_log.txt"
            resolution_log_file = open(log_path, 'w')
            resolution_log_file.write(f"Stage 10 Resolution Log for Content ID: {content_id}\n")
            resolution_log_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            resolution_log_file.write("=" * 80 + "\n\n")
            result['data']['output_files'].append(str(log_path))
        
        # Initialize resolvers
        embedding_resolver = None
        if speaker_centroids and initial_embedding_count > 0:
            # Only initialize if we have NEEDS_EMBEDDING segments
            # Use audio_path if provided, otherwise will try S3
            audio_path_str = str(audio_path) if audio_path else None
            embedding_resolver = EmbeddingResolver(
                speaker_centroids,
                similarity_threshold=0.5,
                audio_path=audio_path_str,
                s3_storage=s3_storage
            )
            if audio_path:
                logger.debug(f"[{content_id}] Initialized audio embedding resolver with local audio: {audio_path}")
            else:
                logger.debug(f"[{content_id}] Initialized audio embedding resolver (will download from S3)")
            logger.debug(f"[{content_id}] Using {len(speaker_centroids)} speaker centroids")
        elif initial_embedding_count > 0:
            logger.warning(f"[{content_id}] No speaker centroids provided - cannot resolve NEEDS_EMBEDDING segments")
        
        # Only initialize LLM resolver if there are segments that need it
        llm_resolver = None
        if initial_llm_count > 0 or initial_unknown_count > 0:
            llm_resolver = Stage10LLMResolver()
            logger.debug(f"[{content_id}] Initialized LLM resolver for {initial_llm_count + initial_unknown_count} segments")
        else:
            logger.info(f"[{content_id}] No NEEDS_LLM or UNKNOWN segments - skipping LLM resolver initialization")
        
        # Identify all unresolved segments
        unresolved_segments = identify_unresolved_segments(word_table, target_labels)
        logger.info(f"[{content_id}] Found {len(unresolved_segments)} unresolved segments")
        
        # Process segments in three passes: simple resolution, embeddings, then LLM
        total_simple_resolved = 0
        total_embedding_resolved = 0
        total_llm_resolved = 0
        resolution_results = []
        
        # PASS 1: Try simple resolution first (most efficient) for ALL unresolved segments
        simple_resolution_count = 0
        simple_resolved_by_type = {'NEEDS_EMBEDDING': 0, 'NEEDS_LLM': 0, 'UNKNOWN': 0}
        
        if unresolved_segments:
            logger.info(f"[{content_id}] PASS 1: Attempting simple resolution for all {len(unresolved_segments)} unresolved segments")
            
            if resolution_log_file:
                resolution_log_file.write(f"\nPASS 1: SIMPLE RESOLUTION\n")
                resolution_log_file.write(f"Attempting simple resolution for all {len(unresolved_segments)} segments\n")
                resolution_log_file.write(f"(NEEDS_EMBEDDING: {sum(1 for s in unresolved_segments if s['label'] == 'NEEDS_EMBEDDING')}, "
                                        f"NEEDS_LLM: {sum(1 for s in unresolved_segments if s['label'] == 'NEEDS_LLM')}, "
                                        f"UNKNOWN: {sum(1 for s in unresolved_segments if s['label'] == 'UNKNOWN')})\n")
                resolution_log_file.write("=" * 80 + "\n")
            
            for i, segment in enumerate(unresolved_segments):
                simple_speaker = try_simple_resolution(segment, word_table, diarization_data)
                
                if simple_speaker:
                    original_label = segment['label']
                    logger.info(f"[{content_id}] Simple resolution: {original_label} segment with {segment['word_count']} words '{segment['text'][:50]}...' -> {simple_speaker}")
                    
                    # Apply the simple resolution
                    for word_info in segment['words']:
                        update_word_assignment(
                            word_table,
                            word_info['word_idx'],
                            'stage10_simple_resolution',
                            speaker=simple_speaker,
                            confidence=0.95,
                            method='simple_heuristics',
                            metadata={
                                'resolution_type': 'simple_diarization' if segment['word_count'] > 1 else 'single_word_context',
                                'original_label': original_label,
                                'segment_text': segment['text'][:100],
                                'reason': 'Resolved using simple heuristics based on diarization overlap or context'
                            }
                        )
                        total_simple_resolved += 1
                    
                    # Mark segment as resolved
                    segment['resolved'] = True
                    simple_resolution_count += 1
                    simple_resolved_by_type[original_label] = simple_resolved_by_type.get(original_label, 0) + 1
                    
                    resolution_results.append({
                        'segment_index': i,
                        'original_label': original_label,
                        'segment_text': segment['text'],
                        'word_count': segment['word_count'],
                        'resolved': True,
                        'method': 'simple_resolution',
                        'speaker': simple_speaker
                    })
                    
                    if resolution_log_file:
                        resolution_log_file.write(f"\nSegment {i+1}: RESOLVED to {simple_speaker}\n")
                        resolution_log_file.write(f"Original Label: {original_label}\n")
                        resolution_log_file.write(f"Text: \"{segment['text'][:100]}{'...' if len(segment['text']) > 100 else ''}\"\n")
                        resolution_log_file.write(f"Method: {'All diarization same speaker' if segment['word_count'] > 1 else 'Single word between same speaker'}\n")
                        resolution_log_file.write("=" * 40 + "\n")
            
            logger.info(f"[{content_id}] Simple resolution resolved {simple_resolution_count}/{len(unresolved_segments)} segments ({total_simple_resolved} words)")
            logger.info(f"[{content_id}] By type - NEEDS_EMBEDDING: {simple_resolved_by_type['NEEDS_EMBEDDING']}, "
                       f"NEEDS_LLM: {simple_resolved_by_type['NEEDS_LLM']}, UNKNOWN: {simple_resolved_by_type['UNKNOWN']}")
        
        # Convert short NEEDS_EMBEDDING segments to NEEDS_LLM after simple resolution
        # First identify remaining NEEDS_EMBEDDING segments
        remaining_embedding_segments = [seg for seg in unresolved_segments if seg['label'] == 'NEEDS_EMBEDDING' and not seg.get('resolved', False)]
        short_embedding_count = 0
        
        for segment in remaining_embedding_segments:
            duration = segment['end_time'] - segment['start_time']
            if duration < 0.75:  # Less than 0.75 seconds
                logger.info(f"[{content_id}] Converting short NEEDS_EMBEDDING segment ({duration:.2f}s) to NEEDS_LLM: '{segment['text']}'")
                # Convert all words in this segment to NEEDS_LLM
                for word_info in segment['words']:
                    word_idx = word_info['word_idx']
                    word_table.df.at[word_idx, 'speaker_current'] = 'NEEDS_LLM'
                    
                    # Update assignment history
                    history = word_table.df.at[word_idx, 'assignment_history']
                    if not isinstance(history, list):
                        history = []
                    history.append({
                        'stage': 'stage10_short_embedding_conversion',
                        'timestamp': time.time(),
                        'speaker': 'NEEDS_LLM',
                        'method': 'short_segment_to_llm',
                        'confidence': 0.0,
                        'reason': f'NEEDS_EMBEDDING segment too short ({duration:.2f}s < 0.75s) for reliable embedding'
                    })
                    word_table.df.at[word_idx, 'assignment_history'] = history
                short_embedding_count += len(segment['words'])
                # Update the segment label for later processing
                segment['label'] = 'NEEDS_LLM'
        
        if short_embedding_count > 0:
            logger.info(f"[{content_id}] Converted {short_embedding_count} words from short NEEDS_EMBEDDING segments to NEEDS_LLM")
        
        # PASS 2: Process remaining NEEDS_EMBEDDING segments
        embedding_segments = [seg for seg in unresolved_segments if seg['label'] == 'NEEDS_EMBEDDING' and not seg.get('resolved', False)]
        if embedding_segments and embedding_resolver:
            logger.info(f"[{content_id}] PASS 2: Processing {len(embedding_segments)} NEEDS_EMBEDDING segments")
            
            if resolution_log_file:
                resolution_log_file.write(f"\nPASS 2: AUDIO EMBEDDING RESOLUTION\n")
                resolution_log_file.write(f"Processing {len(embedding_segments)} NEEDS_EMBEDDING segments\n")
                resolution_log_file.write("=" * 80 + "\n")
            
            for i, segment in enumerate(embedding_segments):
                try:
                    segment_idx = next(idx for idx, seg in enumerate(unresolved_segments) if seg == segment)
                    
                    logger.debug(f"[{content_id}] Processing segment {segment_idx}: "
                              f"{segment['word_count']} words at {segment['start_time']:.2f}s")
                    
                    if resolution_log_file:
                        resolution_log_file.write(f"\nEMBEDDING SEGMENT {i+1}/{len(embedding_segments)}\n")
                        resolution_log_file.write(f"Words: {segment['word_count']}\n")
                        resolution_log_file.write(f"Text: \"{segment['text']}\"\n")
                        resolution_log_file.write(f"Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s\n")
                        resolution_log_file.write("-" * 40 + "\n")
                    
                    embedding_result = await embedding_resolver.resolve_segment(segment, content_id)
                    
                    if resolution_log_file:
                        resolution_log_file.write(f"Audio embedding resolution: {embedding_result['status']}\n")
                        if embedding_result.get('similarity', 0) > 0:
                            resolution_log_file.write(f"Best similarity: {embedding_result['similarity']:.3f}\n")
                        if 'reason' in embedding_result:
                            resolution_log_file.write(f"Reason: {embedding_result['reason']}\n")
                    
                    if embedding_result['status'] == 'success':
                        # Apply embedding assignment
                        speaker = embedding_result['speaker']
                        for word_info in segment['words']:
                            update_word_assignment(
                                word_table,
                                word_info['word_idx'],
                                'stage10_audio_embedding_resolution',
                                speaker=speaker,
                                confidence=embedding_result['similarity'],
                                method='audio_embedding_similarity',
                                metadata={
                                    'similarity': embedding_result['similarity'],
                                    'segment_index': segment_idx,
                                    'duration': embedding_result.get('duration', 0),
                                    'reason': f'Matched to {speaker} via audio embedding similarity'
                                }
                            )
                            total_embedding_resolved += 1
                        
                        logger.info(f"[{content_id}] Resolved {len(segment['words'])} words via audio embedding to {speaker}")
                        
                        if resolution_log_file:
                            resolution_log_file.write(f"Resolved to: {speaker}\n")
                        
                        resolution_results.append({
                            'segment_index': segment_idx,
                            'original_label': segment['label'],
                            'segment_text': segment['text'],
                            'word_count': segment['word_count'],
                            'resolved': True,
                            'method': 'audio_embedding'
                        })
                        
                        # Mark segment as resolved
                        segment['resolved'] = True
                        
                    elif embedding_result['status'] == 'below_threshold':
                        # Mark for LLM resolution
                        logger.info(f"[{content_id}] Audio embedding similarity below threshold - will try LLM in pass 2")
                        segment['label'] = 'NEEDS_LLM'  # Convert for LLM processing
                        
                        if resolution_log_file:
                            resolution_log_file.write(f"Below threshold - marked for LLM resolution\n")
                    
                    else:
                        # Failed embedding - will try LLM
                        logger.warning(f"[{content_id}] Audio embedding failed - will try LLM in pass 2")
                        segment['label'] = 'NEEDS_LLM'  # Convert for LLM processing
                        
                        if resolution_log_file:
                            resolution_log_file.write(f"Embedding failed - marked for LLM resolution\n")
                    
                    if resolution_log_file:
                        resolution_log_file.write("=" * 40 + "\n")
                        
                except Exception as e:
                    logger.error(f"[{content_id}] Error processing embedding segment {i+1}: {str(e)}")
                    segment['label'] = 'NEEDS_LLM'  # Convert failed embedding to LLM
                    
                    if resolution_log_file:
                        resolution_log_file.write(f"Error: {str(e)} - marked for LLM resolution\n")
                        resolution_log_file.write("=" * 40 + "\n")
        
        # Clean up embedding resolver resources after all embedding processing is complete
        if embedding_resolver:
            embedding_resolver.cleanup()
            logger.debug(f"[{content_id}] Cleaned up embedding resolver resources")
        
        # PASS 3: Process remaining unresolved segments with LLM
        llm_segments = [seg for seg in unresolved_segments if seg['label'] in ['NEEDS_LLM', 'UNKNOWN'] and not seg.get('resolved', False)]
        if llm_segments:
            logger.info(f"[{content_id}] PASS 3: Processing {len(llm_segments)} remaining segments with LLM")
            
            if resolution_log_file:
                resolution_log_file.write(f"\nPASS 2: LLM RESOLUTION\n")
                resolution_log_file.write(f"Processing {len(llm_segments)} segments requiring LLM\n")
                resolution_log_file.write("=" * 80 + "\n")
            
            # Initialize LLM resolver if needed
            if llm_resolver is None:
                logger.info(f"[{content_id}] Initializing LLM resolver for pass 2")
                llm_resolver = Stage10LLMResolver()
            
            for i, segment in enumerate(llm_segments):
                try:
                    segment_idx = next(idx for idx, seg in enumerate(unresolved_segments) if seg == segment)
                    
                    logger.debug(f"[{content_id}] Processing segment {segment_idx}: "
                              f"{segment['word_count']} words at {segment['start_time']:.2f}s")
                    
                    if resolution_log_file:
                        resolution_log_file.write(f"\nLLM SEGMENT {i+1}/{len(llm_segments)}\n")
                        resolution_log_file.write(f"Original Label: {segment['label']}\n")
                        resolution_log_file.write(f"Words: {segment['word_count']}\n")
                        resolution_log_file.write(f"Text: \"{segment['text']}\"\n")
                        resolution_log_file.write(f"Time: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s\n")
                        resolution_log_file.write("-" * 40 + "\n")
                    
                    # Gather context
                    context = gather_segment_context(segment, word_table, diarization_data)
                    
                    # Resolve with LLM
                    llm_result = await llm_resolver.resolve_unknown_segment(context, test_mode=test_mode)
                    
                    if resolution_log_file:
                        resolution_log_file.write(f"LLM Response: {llm_result.get('llm_response', 'No response')}\n")
                        resolution_log_file.write(f"Status: {llm_result['status']}\n")
                    
                    if llm_result['status'] == 'success' and llm_result['assignments']:
                        # Apply LLM assignments
                        for word_idx, speaker in llm_result['assignments'].items():
                            update_word_assignment(
                                word_table,
                                word_idx,
                                'stage10_llm_resolution',
                                speaker=speaker,
                                confidence=0.80,
                                method='llm_resolution',
                                metadata={
                                    'segment_index': segment_idx,
                                    'word_count': segment['word_count'],
                                    'reason': 'LLM resolution for complex case'
                                }
                            )
                            total_llm_resolved += 1
                        
                        logger.info(f"[{content_id}] Resolved {len(llm_result['assignments'])} words via LLM")
                        
                        if resolution_log_file:
                            resolution_log_file.write(f"Resolved: {len(llm_result['assignments'])} words\n")
                        
                        resolution_results.append({
                            'segment_index': segment_idx,
                            'original_label': segment['label'],
                            'segment_text': segment['text'],
                            'word_count': segment['word_count'],
                            'resolved': True,
                            'method': 'llm'
                        })
                        
                    else:
                        logger.warning(f"[{content_id}] Failed to resolve LLM segment {i+1}")
                        if resolution_log_file:
                            resolution_log_file.write(f"Failed: {llm_result.get('reason', 'Unknown error')}\n")
                        
                        resolution_results.append({
                            'segment_index': segment_idx,
                            'original_label': segment['label'],
                            'segment_text': segment['text'],
                            'word_count': segment['word_count'],
                            'resolved': False,
                            'method': 'llm_failed'
                        })
                    
                    if resolution_log_file:
                        resolution_log_file.write("=" * 40 + "\n")
                        
                except Exception as e:
                    logger.error(f"[{content_id}] Error processing LLM segment {i+1}: {str(e)}")
                    resolution_results.append({
                        'segment_index': segment_idx,
                        'original_label': segment['label'],
                        'segment_text': segment['text'],
                        'word_count': segment['word_count'],
                        'resolved': False,
                        'error': str(e)
                    })
                    
                    if resolution_log_file:
                        resolution_log_file.write(f"Error: {str(e)}\n")
                        resolution_log_file.write("=" * 40 + "\n")
        
        # Add any segments that weren't processed to resolution_results
        for i, segment in enumerate(unresolved_segments):
            if not any(r['segment_index'] == i for r in resolution_results):
                resolution_results.append({
                    'segment_index': i,
                    'original_label': segment['label'],
                    'segment_text': segment['text'],
                    'word_count': segment['word_count'],
                    'resolved': False,
                    'method': 'not_processed'
                })
        
        # Final statistics
        final_embedding_count = len(word_table.df[word_table.df['speaker_current'] == 'NEEDS_EMBEDDING'])
        final_llm_count = len(word_table.df[word_table.df['speaker_current'] == 'NEEDS_LLM'])
        final_unknown_count = len(word_table.df[word_table.df['speaker_current'] == 'UNKNOWN'])
        final_total = final_embedding_count + final_llm_count + final_unknown_count
        
        if resolution_log_file:
            resolution_log_file.write("\n" + "=" * 80 + "\n")
            resolution_log_file.write("STAGE 10 SUMMARY\n")
            resolution_log_file.write("=" * 80 + "\n")
            resolution_log_file.write(f"Segments processed: {len(unresolved_segments)}\n")
            resolution_log_file.write(f"Words resolved via simple heuristics: {total_simple_resolved}\n")
            resolution_log_file.write(f"Words resolved via embedding: {total_embedding_resolved}\n")
            resolution_log_file.write(f"Words resolved via LLM: {total_llm_resolved}\n")
            resolution_log_file.write(f"Initial unresolved: {initial_total}\n")
            resolution_log_file.write(f"Final unresolved: {final_total}\n")
            resolution_log_file.write(f"Reduction: {initial_total - final_total} "
                                    f"({(1 - final_total/initial_total)*100:.1f}% if initial_total > 0 else 100)\n")
            resolution_log_file.write(f"Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            resolution_log_file.close()
        
        result['status'] = 'success'
        result['data']['word_table'] = word_table
        result['data']['resolution_results'] = resolution_results
        result['stats'] = {
            'duration': time.time() - start_time,
            'category_labels_converted': category_count,
            'short_embeddings_converted': short_embedding_count,
            'initial_total': initial_total,
            'initial_embedding': initial_embedding_count,
            'initial_llm': initial_llm_count,
            'initial_unknown': initial_unknown_count,
            'segments_processed': len(unresolved_segments),
            'simple_resolved': total_simple_resolved,
            'simple_resolved_by_type': simple_resolved_by_type,
            'embedding_resolved': total_embedding_resolved,
            'llm_resolved': total_llm_resolved,
            'final_total': final_total,
            'final_embedding': final_embedding_count,
            'final_llm': final_llm_count,
            'final_unknown': final_unknown_count,
            'reduction_percentage': (1 - final_total/initial_total)*100 if initial_total > 0 else 100
        }
        
        logger.info(f"[{content_id}] Stage 10 completed: {total_simple_resolved + total_embedding_resolved + total_llm_resolved} words resolved "
                   f"({total_simple_resolved} via simple, {total_embedding_resolved} via embedding, {total_llm_resolved} via LLM), "
                   f"{final_total} words remain unresolved")
        
        # Add summary of speaker assignments in test mode
        if test_mode:
            summarize_speaker_assignments(word_table, 'stage10_resolutions', content_id, test_mode)
        
        return result
        
    except Exception as e:
        logger.error(f"[{content_id}] Stage 8 failed: {str(e)}")
        logger.error(f"[{content_id}] Error details:", exc_info=True)
        
        if resolution_log_file and not resolution_log_file.closed:
            resolution_log_file.write(f"\nERROR: {str(e)}\n")
            resolution_log_file.close()
        
        # Clean up embedding resolver resources on error
        if 'embedding_resolver' in locals() and embedding_resolver:
            try:
                embedding_resolver.cleanup()
                logger.debug(f"[{content_id}] Cleaned up embedding resolver resources after error")
            except Exception as cleanup_error:
                logger.warning(f"[{content_id}] Failed to cleanup embedding resolver: {cleanup_error}")
        
        result.update({
            'status': 'error',
            'error': str(e),
            'data': {'word_table': word_table},
            'stats': {'duration': time.time() - start_time}
        })
        return result


if __name__ == "__main__":
    # Test execution
    import argparse
    
    parser = argparse.ArgumentParser(description="Stage 8: LLM Resolution for Remaining UNKNOWN Words")
    parser.add_argument("--content", required=True, help="Content ID to process")
    parser.add_argument("--test", action="store_true", help="Run in test mode with detailed logging")
    parser.add_argument("--output-dir", help="Output directory for logs")
    
    args = parser.parse_args()
    
    logger.info(f"Running Stage 8 LLM Resolution for content: {args.content}")
    
    # This would typically load the word table and diarization data
    # For testing, we'll just show the structure
    print(f"Stage 8: LLM Resolution for Remaining UNKNOWN Words")
    print(f"Content ID: {args.content}")
    print(f"Test Mode: {args.test}")
    print(f"Output Dir: {args.output_dir}")
    print("\nThis stage would:")
    print("1. Load the word table from previous stages")
    print("2. Identify remaining UNKNOWN words")
    print("3. Use comprehensive LLM analysis to resolve them")
    print("4. Apply final assignments with appropriate confidence")
    print("5. Generate detailed logs of the resolution process")