#!/usr/bin/env python3
"""
Parakeet Single-Pass Transcription Method

Simple single-pass transcription using Parakeet MLX (v2 for English, v3 for other European languages).
No VAD, no diarization - just direct transcription.

Method: parakeet_single_pass
Requires VAD: False
Requires Diarization: False
"""

import os
import sys
from pathlib import Path
from typing import Dict, Optional
from contextlib import contextmanager
import time

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('parakeet_single_pass')

# Method metadata
METHOD_NAME = "parakeet_single_pass"
REQUIRES_VAD = False
REQUIRES_DIARIZATION = False

# Parameters
MAX_CHUNK_DURATION = 600.0  # 10 minutes

# Supported languages
PARAKEET_V2_LANGUAGES = {'en'}
PARAKEET_V3_LANGUAGES = {
    'bg', 'hr', 'cs', 'da', 'nl', 'et', 'fi', 'fr', 'de',
    'el', 'hu', 'it', 'lv', 'lt', 'mt', 'pl', 'pt', 'ro', 'sk',
    'sl', 'es', 'sv', 'ru', 'uk'
}


@contextmanager
def suppress_stderr():
    """Context manager to temporarily suppress stderr output"""
    with open(os.devnull, 'w') as devnull:
        stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = stderr


def transcribe_with_parakeet(audio_path: str, language: str) -> Dict:
    """
    Run Parakeet MLX transcription (v2 for English, v3 for European languages)

    Args:
        audio_path: Path to audio file
        language: Language code (ISO 639-1 format, e.g., 'en', 'fr')

    Returns:
        Dict with segments, text, language, method
    """
    try:
        with suppress_stderr():
            # Determine which Parakeet version to use
            use_parakeet_v2 = language in PARAKEET_V2_LANGUAGES
            use_parakeet_v3 = language in PARAKEET_V3_LANGUAGES

            if use_parakeet_v2:
                model_name = "mlx-community/parakeet-tdt-0.6b-v2"
                logger.info(f"Using Parakeet MLX v2 for English")

                from parakeet_mlx import from_pretrained
                model = from_pretrained(model_name)

                start_time = time.time()
                result = model.transcribe(audio_path)
                elapsed_time = time.time() - start_time

                logger.info(f"Parakeet v2 transcription completed in {elapsed_time:.2f} seconds")

                # Convert AlignedResult to standard format
                text = getattr(result, 'text', '')
                segments = []

                if hasattr(result, 'sentences') and result.sentences:
                    for sentence in result.sentences:
                        words = []

                        if hasattr(sentence, 'tokens') and sentence.tokens:
                            current_word = None

                            for token in sentence.tokens:
                                if not hasattr(token, 'text') or not hasattr(token, 'start') or not hasattr(token, 'end'):
                                    continue

                                token_text = token.text

                                # Check if this token starts a new word
                                if token_text.startswith(' ') or current_word is None:
                                    if current_word is not None:
                                        words.append(current_word)

                                    current_word = {
                                        'word': token_text.strip(),
                                        'start': float(token.start),
                                        'end': float(token.end),
                                        'probability': 1.0
                                    }
                                else:
                                    # Continue current word (merge subword tokens)
                                    current_word['word'] += token_text
                                    current_word['end'] = float(token.end)

                            if current_word is not None:
                                words.append(current_word)

                        segment = {
                            'text': getattr(sentence, 'text', ''),
                            'start': float(getattr(sentence, 'start', 0.0)),
                            'end': float(getattr(sentence, 'end', 0.0)),
                            'words': words
                        }
                        segments.append(segment)
                else:
                    segments = [{
                        'text': text,
                        'start': 0.0,
                        'end': 0.0,
                        'words': []
                    }]

                return {
                    'text': text,
                    'segments': segments,
                    'language': 'en',
                    'method': 'parakeet_mlx_v2'
                }

            elif use_parakeet_v3:
                model_name = "mlx-community/parakeet-tdt-0.6b-v3"
                logger.info(f"Using Parakeet MLX v3 with auto-detection (expected: {language})")

                from parakeet_mlx import from_pretrained
                model = from_pretrained(model_name)

                start_time = time.time()
                result = model.transcribe(audio_path)
                elapsed_time = time.time() - start_time

                logger.info(f"Parakeet v3 transcription completed in {elapsed_time:.2f} seconds")

                # Convert AlignedResult to standard format
                text = getattr(result, 'text', '')
                segments = []

                if hasattr(result, 'sentences') and result.sentences:
                    for sentence in result.sentences:
                        words = []

                        if hasattr(sentence, 'tokens') and sentence.tokens:
                            current_word = None

                            for token in sentence.tokens:
                                if not hasattr(token, 'text') or not hasattr(token, 'start') or not hasattr(token, 'end'):
                                    continue

                                token_text = token.text

                                if token_text.startswith(' ') or current_word is None:
                                    if current_word is not None:
                                        words.append(current_word)

                                    current_word = {
                                        'word': token_text.strip(),
                                        'start': float(token.start),
                                        'end': float(token.end),
                                        'probability': 1.0
                                    }
                                else:
                                    current_word['word'] += token_text
                                    current_word['end'] = float(token.end)

                            if current_word is not None:
                                words.append(current_word)

                        segment = {
                            'text': getattr(sentence, 'text', ''),
                            'start': float(getattr(sentence, 'start', 0.0)),
                            'end': float(getattr(sentence, 'end', 0.0)),
                            'words': words
                        }
                        segments.append(segment)
                else:
                    segments = [{
                        'text': text,
                        'start': 0.0,
                        'end': 0.0,
                        'words': []
                    }]

                detected_language = getattr(result, 'language', language)

                return {
                    'text': text,
                    'segments': segments,
                    'language': detected_language,
                    'method': 'parakeet_mlx_v3'
                }

            else:
                # Fall back to Whisper for unsupported languages
                logger.info(f"Language '{language}' not supported by Parakeet, falling back to Whisper")
                import mlx_whisper

                start_time = time.time()
                result = mlx_whisper.transcribe(
                    audio_path,
                    path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                    word_timestamps=True
                )
                elapsed_time = time.time() - start_time
                logger.info(f"Whisper fallback completed in {elapsed_time:.2f} seconds")

                return result

    except Exception as e:
        logger.error(f"Parakeet transcription failed: {e}")
        return {'error': str(e)}


def main(audio_path: str, language: str = 'en',
         vad_json_path: Optional[str] = None,
         diarization_json_path: Optional[str] = None) -> Dict:
    """
    Main entry point for parakeet single-pass transcription.

    Args:
        audio_path: Path to audio file
        language: Language code (default 'en')
        vad_json_path: Not used (for interface compatibility)
        diarization_json_path: Not used (for interface compatibility)

    Returns:
        Dict with:
            - segments: List of transcript segments with word-level timestamps
            - language: Detected language
            - method: Method name
            - stats: Processing statistics
    """
    logger.info(f"Starting Parakeet single-pass transcription")
    logger.info(f"  Audio: {audio_path}")
    logger.info(f"  Language: {language}")

    start_time = time.time()

    # Run transcription
    result = transcribe_with_parakeet(audio_path, language)

    if 'error' in result:
        return result

    processing_time = time.time() - start_time

    # Count total words
    total_words = 0
    for segment in result.get('segments', []):
        if 'words' in segment:
            total_words += len(segment['words'])

    # Add stats
    result['stats'] = {
        'total_segments': len(result.get('segments', [])),
        'total_words': total_words,
        'processing_time': processing_time
    }

    logger.info(f"Parakeet transcription complete:")
    logger.info(f"  Segments: {result['stats']['total_segments']}")
    logger.info(f"  Words: {result['stats']['total_words']}")
    logger.info(f"  Time: {processing_time:.1f}s")

    return result
