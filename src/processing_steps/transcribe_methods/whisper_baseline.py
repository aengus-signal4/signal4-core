#!/usr/bin/env python3
"""
Whisper Baseline Transcription Method

Simple Whisper Large v3 transcription with automatic chunking for long audio.
No VAD, no diarization - just direct transcription.

NOTE: Uses soundfile instead of torchaudio to avoid segfaults caused by
torch/MLX Metal backend conflicts when both are loaded simultaneously.
"""

import os
import sys
import tempfile
import shutil
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import mlx_whisper
import soundfile as sf
import numpy as np

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('whisper_baseline')

# Method metadata
METHOD_NAME = "whisper_mlx_turbo"
REQUIRES_VAD = False
REQUIRES_DIARIZATION = False

# Method-specific parameters (tune here)
MAX_CHUNK_DURATION = 300.0  # Maximum chunk duration in seconds (5 minutes)


def _run_mlx_transcribe(audio_path: str, language: Optional[str] = None) -> Dict:
    """Run MLX Whisper transcription with automatic chunking

    Args:
        audio_path: Path to audio file
        language: Optional language code (e.g., 'en', 'fr')

    Returns:
        Dict with transcription result or error
    """
    try:
        # Use soundfile to load audio (avoids torch/MLX conflicts)
        audio, sample_rate = sf.read(audio_path)

        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        audio_duration = len(audio) / sample_rate

        # If audio is short enough, transcribe directly
        if audio_duration <= MAX_CHUNK_DURATION:
            logger.info(f"Audio duration {audio_duration:.1f}s <= {MAX_CHUNK_DURATION}s, transcribing directly")

            # Prepare transcription kwargs
            transcribe_kwargs = {
                "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
                "word_timestamps": True
            }

            if language:
                transcribe_kwargs["language"] = language
                logger.info(f"Using specified language: {language}")

            # Run transcription
            start_time = time.time()
            result = mlx_whisper.transcribe(audio_path, **transcribe_kwargs)
            elapsed_time = time.time() - start_time
            logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")

            return result

        # Audio is too long, need to chunk it
        logger.info(f"Audio duration {audio_duration:.1f}s > {MAX_CHUNK_DURATION}s, chunking required")

        # Create temp directory for chunks
        temp_dir = Path(tempfile.mkdtemp(prefix="whisper_chunks_"))
        try:
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                from scipy import signal as scipy_signal
                num_samples = int(len(audio) * 16000 / sample_rate)
                audio = scipy_signal.resample(audio, num_samples)
                sample_rate = 16000

            # Split into chunks
            chunk_samples = int(MAX_CHUNK_DURATION * sample_rate)
            num_chunks = int(np.ceil(len(audio) / chunk_samples))

            logger.info(f"Splitting into {num_chunks} chunks of max {MAX_CHUNK_DURATION}s")

            all_segments = []
            current_offset = 0.0

            for chunk_idx in range(num_chunks):
                start_sample = chunk_idx * chunk_samples
                end_sample = min((chunk_idx + 1) * chunk_samples, len(audio))

                chunk_audio = audio[start_sample:end_sample]
                chunk_duration = len(chunk_audio) / sample_rate

                # Save chunk
                chunk_path = temp_dir / f"chunk_{chunk_idx}.wav"
                sf.write(str(chunk_path), chunk_audio, sample_rate)

                logger.info(f"Transcribing chunk {chunk_idx + 1}/{num_chunks} ({chunk_duration:.1f}s)")

                # Transcribe chunk
                transcribe_kwargs = {
                    "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
                    "word_timestamps": True
                }

                if language:
                    transcribe_kwargs["language"] = language

                chunk_result = mlx_whisper.transcribe(str(chunk_path), **transcribe_kwargs)

                # Adjust timestamps
                for segment in chunk_result.get('segments', []):
                    segment['start'] = segment.get('start', 0) + current_offset
                    segment['end'] = segment.get('end', 0) + current_offset

                    if 'words' in segment:
                        for word in segment['words']:
                            word['start'] = word.get('start', 0) + current_offset
                            word['end'] = word.get('end', 0) + current_offset

                    all_segments.append(segment)

                current_offset += chunk_duration

            # Combine results
            full_text = ' '.join(seg.get('text', '') for seg in all_segments)

            result = {
                'text': full_text,
                'segments': all_segments,
                'language': all_segments[0].get('language', 'en') if all_segments else 'en'
            }

            logger.info(f"Chunked transcription complete: {len(all_segments)} segments")
            return result

        finally:
            # Clean up temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        logger.error(traceback.format_exc())
        return {'error': str(e)}


def main(
    audio_path: str,
    language: str = 'en',
    vad_json_path: Optional[str] = None,
    diarization_json_path: Optional[str] = None
) -> Dict:
    """
    Main entry point for Whisper baseline transcription.

    Args:
        audio_path: Path to audio file
        language: Language code (e.g., 'en', 'fr')
        vad_json_path: Ignored (method doesn't use VAD)
        diarization_json_path: Ignored (method doesn't use diarization)

    Returns:
        Dict with keys:
            - segments: List of transcription segments with word timestamps
            - language: Detected/specified language
            - method: 'whisper_baseline'
            - stats: Processing statistics
    """
    logger.info("="*80)
    logger.info("Whisper Baseline Transcription")
    logger.info("="*80)
    logger.info(f"Audio: {audio_path}")
    logger.info(f"Language: {language}")

    start_time = time.time()
    result = _run_mlx_transcribe(audio_path, language=language)
    processing_time = time.time() - start_time

    if 'error' in result:
        return {
            'error': result['error'],
            'method': METHOD_NAME
        }

    # Calculate stats
    segments = result.get('segments', [])
    total_words = sum(len(seg.get('words', [])) for seg in segments)

    return {
        'segments': segments,
        'language': language,  # Trust specified language over auto-detection
        'method': METHOD_NAME,
        'stats': {
            'total_segments': len(segments),
            'total_words': total_words,
            'processing_time': processing_time
        }
    }
