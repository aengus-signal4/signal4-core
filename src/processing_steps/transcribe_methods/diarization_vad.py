#!/usr/bin/env python3
"""
Diarization-VAD Transcription Method

Uses diarization segments as natural speaker turn boundaries, expands them with VAD
for complete speech coverage, and transcribes each segment individually.

Method: diarization_vad
Requires VAD: True
Requires Diarization: True
"""

import json
import numpy as np
import soundfile as sf
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('diarization_vad')

# Method metadata
METHOD_NAME = "diarization_vad"
REQUIRES_VAD = True
REQUIRES_DIARIZATION = True

# Parameters
MAX_EXPANSION = 0.5  # Maximum expansion beyond diarization boundaries (seconds)


@dataclass
class DiarizationSegment:
    """A speaker turn from diarization"""
    start: float
    end: float
    speaker: str
    duration: float


@dataclass
class VADSegment:
    """A speech segment from VAD"""
    start: float
    end: float
    duration: float
    vad_index: int


@dataclass
class TranscriptionSegment:
    """An expanded segment ready for transcription"""
    diar_start: float
    diar_end: float
    speaker: str
    vad_start: float
    vad_end: float
    vad_indices: List[int]
    duration: float


def load_parakeet_model():
    """Load Parakeet v3 model"""
    from parakeet_mlx import from_pretrained
    return from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")


def expand_diarization_with_vad(
    diar_segments: List[DiarizationSegment],
    vad_segments: List[VADSegment]
) -> List[TranscriptionSegment]:
    """
    Expand diarization segments to cover overlapping VAD segments

    Args:
        diar_segments: Speaker turns from diarization
        vad_segments: Speech segments from VAD

    Returns:
        List of expanded segments ready for transcription
    """
    transcription_segments = []

    for diar in diar_segments:
        # Find VAD segments that overlap with this diarization segment
        overlapping_vad = []

        for vad in vad_segments:
            overlap_start = max(diar.start, vad.start)
            overlap_end = min(diar.end, vad.end)

            if overlap_end > overlap_start:
                overlapping_vad.append(vad)
            elif vad.start > diar.end + MAX_EXPANSION:
                break

        if not overlapping_vad:
            # No VAD overlap - use diarization boundaries as-is
            transcription_segments.append(TranscriptionSegment(
                diar_start=diar.start,
                diar_end=diar.end,
                speaker=diar.speaker,
                vad_start=diar.start,
                vad_end=diar.end,
                vad_indices=[],
                duration=diar.duration
            ))
            continue

        # Expand to cover all overlapping VAD segments
        vad_start = min(v.start for v in overlapping_vad)
        vad_end = max(v.end for v in overlapping_vad)

        # Limit expansion
        vad_start = max(vad_start, diar.start - MAX_EXPANSION)
        vad_end = min(vad_end, diar.end + MAX_EXPANSION)

        transcription_segments.append(TranscriptionSegment(
            diar_start=diar.start,
            diar_end=diar.end,
            speaker=diar.speaker,
            vad_start=vad_start,
            vad_end=vad_end,
            vad_indices=[v.vad_index for v in overlapping_vad],
            duration=vad_end - vad_start
        ))

    return transcription_segments


def transcribe_segment(
    model,
    audio: np.ndarray,
    sr: int,
    segment: TranscriptionSegment,
    temp_dir: Path
) -> Dict:
    """
    Transcribe a single expanded diarization segment

    Returns dict with:
        - status: 'success' or 'error'
        - words: List of word-level timestamps (if success)
        - text: Full text (if success)
    """
    try:
        # Extract audio for this segment
        start_sample = int(segment.vad_start * sr)
        end_sample = int(segment.vad_end * sr)
        segment_audio = audio[start_sample:end_sample]

        # Save to temp file
        temp_file = temp_dir / f"segment_{segment.diar_start:.2f}_{segment.diar_end:.2f}.wav"
        sf.write(str(temp_file), segment_audio, sr)

        # Transcribe
        result = model.transcribe(str(temp_file))

        # Extract words with timestamps
        words = []

        if hasattr(result, 'sentences') and result.sentences:
            for sentence in result.sentences:
                if hasattr(sentence, 'tokens') and sentence.tokens:
                    current_word = None

                    for token in sentence.tokens:
                        if not hasattr(token, 'text') or not hasattr(token, 'start') or not hasattr(token, 'end'):
                            continue

                        token_text = token.text

                        # Merge tokens into words
                        if token_text.startswith(' ') or current_word is None:
                            if current_word is not None:
                                words.append({
                                    'word': current_word['word'],
                                    'start': current_word['start'] + segment.vad_start,
                                    'end': current_word['end'] + segment.vad_start,
                                    'speaker': segment.speaker
                                })

                            current_word = {
                                'word': token_text.strip(),
                                'start': float(token.start),
                                'end': float(token.end)
                            }
                        else:
                            current_word['word'] += token_text
                            current_word['end'] = float(token.end)

                    if current_word is not None:
                        words.append({
                            'word': current_word['word'],
                            'start': current_word['start'] + segment.vad_start,
                            'end': current_word['end'] + segment.vad_start,
                            'speaker': segment.speaker
                        })

        # Clean up temp file
        temp_file.unlink()

        return {
            'status': 'success',
            'words': words,
            'text': result.text if hasattr(result, 'text') else '',
            'speaker': segment.speaker,
            'start': segment.diar_start,
            'end': segment.diar_end
        }

    except Exception as e:
        logger.error(f"Error transcribing segment {segment.diar_start:.2f}-{segment.diar_end:.2f}s: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'speaker': segment.speaker,
            'start': segment.diar_start,
            'end': segment.diar_end
        }


def main(audio_path: str, language: str = 'en',
         vad_json_path: Optional[str] = None,
         diarization_json_path: Optional[str] = None) -> Dict:
    """
    Main entry point for diarization-VAD transcription.

    Args:
        audio_path: Path to audio file
        language: Language code (default 'en')
        vad_json_path: Path to VAD JSON (required)
        diarization_json_path: Path to diarization JSON (required)

    Returns:
        Dict with:
            - segments: List of transcript segments with speaker attribution
            - language: Language code
            - method: Method name
            - stats: Processing statistics
    """
    if not vad_json_path:
        return {'error': 'VAD JSON required for diarization_vad method'}

    if not diarization_json_path:
        return {'error': 'Diarization JSON required for diarization_vad method'}

    logger.info(f"Starting diarization-VAD transcription")
    logger.info(f"  Audio: {audio_path}")
    logger.info(f"  VAD: {vad_json_path}")
    logger.info(f"  Diarization: {diarization_json_path}")

    # Load audio
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    audio_duration = len(audio) / sr
    logger.info(f"  Audio duration: {audio_duration:.2f}s")

    # Load diarization
    with open(diarization_json_path) as f:
        diar_data = json.load(f)

    diar_segments = []
    for seg in diar_data['segments']:
        if seg['end'] > 0 and seg['start'] < audio_duration:
            diar_segments.append(DiarizationSegment(
                start=max(0, seg['start']),
                end=min(audio_duration, seg['end']),
                speaker=seg['speaker'],
                duration=seg['end'] - seg['start']
            ))

    logger.info(f"  Diarization segments: {len(diar_segments)}")

    # Load VAD
    with open(vad_json_path) as f:
        vad_data = json.load(f)

    vad_segments = [
        VADSegment(
            start=seg['start'],
            end=seg['end'],
            duration=seg.get('duration', seg['end'] - seg['start']),
            vad_index=i
        )
        for i, seg in enumerate(vad_data['segments'])
    ]

    logger.info(f"  VAD segments: {len(vad_segments)}")

    # Expand diarization with VAD
    transcription_segments = expand_diarization_with_vad(diar_segments, vad_segments)
    logger.info(f"  Transcription segments: {len(transcription_segments)}")

    # Load Parakeet model
    logger.info("Loading Parakeet v3 model...")
    model = load_parakeet_model()

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="transcribe_diar_"))

    try:
        # Transcribe each segment
        all_words = []
        segment_results = []

        for i, seg in enumerate(transcription_segments):
            logger.info(f"  [{i+1}/{len(transcription_segments)}] Transcribing {seg.speaker} {seg.vad_start:.2f}s-{seg.vad_end:.2f}s ({seg.duration:.2f}s)")

            result = transcribe_segment(model, audio, sr, seg, temp_dir)
            segment_results.append(result)

            if result['status'] == 'success':
                all_words.extend(result['words'])
                logger.info(f"       ✓ {len(result['words'])} words")
            else:
                logger.error(f"       ✗ Error: {result.get('error', 'unknown')}")

        # Convert to standard format
        segments = []
        for result in segment_results:
            if result['status'] == 'success':
                segments.append({
                    'text': result['text'],
                    'start': result['start'],
                    'end': result['end'],
                    'speaker': result['speaker'],
                    'words': [w for w in all_words if w['start'] >= result['start'] and w['end'] <= result['end']]
                })

        # Calculate stats
        successful = sum(1 for r in segment_results if r['status'] == 'success')
        failed = sum(1 for r in segment_results if r['status'] == 'error')

        logger.info(f"Transcription complete:")
        logger.info(f"  Segments: {successful}/{len(transcription_segments)}")
        logger.info(f"  Words: {len(all_words)}")

        return {
            'segments': segments,
            'language': language,
            'method': METHOD_NAME,
            'stats': {
                'total_segments': len(transcription_segments),
                'successful': successful,
                'failed': failed,
                'total_words': len(all_words)
            }
        }

    finally:
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
