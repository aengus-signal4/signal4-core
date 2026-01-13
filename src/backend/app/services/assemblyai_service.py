"""
AssemblyAI Transcription Service
=================================

Service for transcribing audio segments using AssemblyAI API.
Accepts pre-extracted audio files for transcription with translation support.

Note: Audio extraction is handled by the media router. This service only
handles transcription of already-extracted audio files.
"""

import os
import time
import tempfile
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import assemblyai as aai

from ..utils.backend_logger import get_logger

logger = get_logger("assemblyai_service")


class AssemblyAIService:
    """Service for AssemblyAI transcription operations"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize AssemblyAI service

        Args:
            api_key: AssemblyAI API key (reads from env if not provided)
        """
        self.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            raise ValueError("AssemblyAI API key not found. Set ASSEMBLYAI_API_KEY environment variable.")

        # Configure AssemblyAI SDK
        aai.settings.api_key = self.api_key
        logger.info("AssemblyAI service initialized")


    def transcribe_from_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        speaker_labels: bool = False,
        word_boost: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_translation: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe audio from local file with optional translation

        Args:
            audio_path: Path to local audio file
            language: Language code (e.g., 'fr', 'en', 'es') - required for translation
            speaker_labels: Enable speaker diarization
            word_boost: List of words to boost
            config: Additional configuration
            enable_translation: If True, request translations to en and fr

        Returns:
            Dict with transcription results including:
            - transcription_text: Original transcription
            - translation_en: English translation (if source not English)
            - translation_fr: French translation (if source not French)
            - language: Formatted language [xx]
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing audio file with AssemblyAI: {audio_path}")

        start_time = time.time()

        try:
            # Build transcription config with translation if enabled
            config_params = {
                "language_code": language,
                "speaker_labels": speaker_labels,
                "word_boost": word_boost or [],
                **(config or {})
            }

            # Add translation targets if enabled and language is specified
            if enable_translation and language:
                # Always request both English and French translations
                # AssemblyAI will skip translating to the source language
                config_params["speech_understanding"] = aai.SpeechUnderstandingRequest(
                    request=aai.SpeechUnderstandingFeatureRequests(
                        translation=aai.TranslationRequest(
                            target_languages=["en", "fr"]
                        )
                    )
                )
                logger.info(f"Translation enabled: source={language}, targets=[en, fr]")

            transcriber_config = aai.TranscriptionConfig(**config_params)

            # Create transcriber and transcribe directly from file path
            transcriber = aai.Transcriber(config=transcriber_config)
            transcript = transcriber.transcribe(audio_path)

            # Wait for completion
            if transcript.status == aai.TranscriptStatus.error:
                raise Exception(f"AssemblyAI transcription failed: {transcript.error}")

            processing_time = time.time() - start_time
            logger.info(f"AssemblyAI transcription completed in {processing_time:.2f}s")

            # Extract results
            detected_language = getattr(transcript, 'language_code', language)
            # Format language as [xx] for display (e.g., [en], [fr], [ro])
            formatted_language = f"[{detected_language}]" if detected_language else None

            result = {
                "transcription_text": transcript.text or "",
                "confidence": transcript.confidence if hasattr(transcript, 'confidence') else None,
                "word_timings": self._extract_word_timings(transcript),
                "speaker_labels": self._extract_speaker_labels(transcript) if speaker_labels else None,
                "audio_duration": transcript.audio_duration / 1000.0 if transcript.audio_duration else None,  # Convert ms to seconds
                "processing_time": processing_time,
                "language": formatted_language,  # Formatted language for frontend display
                "metadata": {
                    "transcript_id": transcript.id,
                    "language_code": detected_language,  # Raw language code for backend
                    "audio_url": getattr(transcript, 'audio_url', None),
                    "status": str(transcript.status),
                }
            }

            # Extract translations if available
            # AssemblyAI returns translations in the top-level `translated_texts` attribute as a dict
            result["translation_en"] = None
            result["translation_fr"] = None

            if hasattr(transcript, 'translated_texts') and transcript.translated_texts:
                logger.info(f"Found translated_texts: {transcript.translated_texts}")
                # translated_texts is a dictionary with language codes as keys
                translations = transcript.translated_texts
                if isinstance(translations, dict):
                    result["translation_en"] = translations.get('en')
                    result["translation_fr"] = translations.get('fr')

                logger.info(f"Translations extracted: en={bool(result['translation_en'])}, fr={bool(result['translation_fr'])}")
            else:
                logger.info("No translated_texts found in response")

            # Calculate API cost (AssemblyAI pricing: ~$0.00025/second for best model)
            if result["audio_duration"]:
                result["api_cost"] = result["audio_duration"] * 0.00025

            return result

        except Exception as e:
            logger.error(f"AssemblyAI transcription error: {e}", exc_info=True)
            raise

    def _extract_word_timings(self, transcript: aai.Transcript) -> Optional[list]:
        """Extract word-level timings from transcript"""
        if not transcript.words:
            return None

        word_timings = []
        for word in transcript.words:
            word_timings.append({
                "word": word.text,
                "start": word.start / 1000.0,  # Convert ms to seconds
                "end": word.end / 1000.0,
                "confidence": word.confidence
            })

        return word_timings

    def _extract_speaker_labels(self, transcript: aai.Transcript) -> Optional[list]:
        """Extract speaker labels from transcript"""
        if not hasattr(transcript, 'utterances') or not transcript.utterances:
            return None

        speaker_labels = []
        for utterance in transcript.utterances:
            speaker_labels.append({
                "speaker": utterance.speaker,
                "start": utterance.start / 1000.0,  # Convert ms to seconds
                "end": utterance.end / 1000.0,
                "text": utterance.text,
                "confidence": utterance.confidence
            })

        return speaker_labels

    def cleanup_temp_file(self, file_path: str):
        """Clean up temporary audio file"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {file_path}: {e}")
