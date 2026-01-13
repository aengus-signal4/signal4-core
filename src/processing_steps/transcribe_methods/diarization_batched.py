#!/usr/bin/env python3
"""
Diarization-Batched Transcription Method

Uses diarization segments with VAD expansion and processes multiple segments
in batches to optimize inference performance.

Method: diarization_batched
Requires VAD: True
Requires Diarization: True
"""

import json
import numpy as np
import soundfile as sf
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import tempfile
import shutil

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('diarization_batched')

# Method metadata
METHOD_NAME = "diarization_batched"
REQUIRES_VAD = True
REQUIRES_DIARIZATION = True

# Parameters
MAX_EXPANSION = 0.5  # Maximum expansion beyond diarization boundaries (seconds)
BATCH_SIZE = 16  # Number of segments to process per batch


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
    batch_id: int = 0


def load_parakeet_model():
    """Load Parakeet v3 model"""
    from parakeet_mlx import from_pretrained
    return from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")


def expand_diarization_with_vad(
    diar_segments: List[DiarizationSegment],
    vad_segments: List[VADSegment]
) -> List[TranscriptionSegment]:
    """Expand diarization segments to cover overlapping VAD segments"""
    transcription_segments = []

    for diar in diar_segments:
        overlapping_vad = []
        for vad in vad_segments:
            overlap_start = max(diar.start, vad.start)
            overlap_end = min(diar.end, vad.end)
            if overlap_end > overlap_start:
                overlapping_vad.append(vad)
            elif vad.start > diar.end + MAX_EXPANSION:
                break

        if not overlapping_vad:
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

        vad_start = min(v.start for v in overlapping_vad)
        vad_end = max(v.end for v in overlapping_vad)
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


def create_batches(segments: List[TranscriptionSegment], batch_size: int) -> List[List[TranscriptionSegment]]:
    """Group segments into batches for parallel processing"""
    batches = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i + batch_size]
        for seg in batch:
            seg.batch_id = len(batches)
        batches.append(batch)
    return batches


def extract_segment_audio(audio: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    """Extract audio segment"""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return audio[start_sample:end_sample]


def transcribe_batch(
    model,
    audio: np.ndarray,
    sr: int,
    batch: List[TranscriptionSegment],
    temp_dir: Path
) -> List[Dict]:
    """
    Transcribe a batch of segments

    Note: True batch inference requires lower-level API access.
    This optimizes by keeping model in memory and minimizing overhead.
    """
    results = []

    for seg in batch:
        try:
            # Extract segment audio
            segment_audio = extract_segment_audio(audio, sr, seg.vad_start, seg.vad_end)

            # Save to temp file
            temp_file = temp_dir / f"seg_{seg.batch_id}_{seg.diar_start:.2f}.wav"
            sf.write(str(temp_file), segment_audio, sr)

            # Transcribe
            result = model.transcribe(str(temp_file))

            # Extract words
            words = []
            if hasattr(result, 'sentences') and result.sentences:
                for sentence in result.sentences:
                    if hasattr(sentence, 'tokens') and sentence.tokens:
                        current_word = None

                        for token in sentence.tokens:
                            if not hasattr(token, 'text') or not hasattr(token, 'start') or not hasattr(token, 'end'):
                                continue

                            token_text = token.text

                            if token_text.startswith(' ') or current_word is None:
                                if current_word is not None:
                                    words.append({
                                        'word': current_word['word'],
                                        'start': current_word['start'] + seg.vad_start,
                                        'end': current_word['end'] + seg.vad_start,
                                        'speaker': seg.speaker
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
                                'start': current_word['start'] + seg.vad_start,
                                'end': current_word['end'] + seg.vad_start,
                                'speaker': seg.speaker
                            })

            # Clean up temp file
            temp_file.unlink()

            results.append({
                'status': 'success',
                'words': words,
                'text': result.text if hasattr(result, 'text') else '',
                'speaker': seg.speaker,
                'start': seg.diar_start,
                'end': seg.diar_end
            })

        except Exception as e:
            logger.error(f"Error transcribing segment {seg.diar_start:.2f}-{seg.diar_end:.2f}s: {e}")
            results.append({
                'status': 'error',
                'error': str(e),
                'speaker': seg.speaker,
                'start': seg.diar_start,
                'end': seg.diar_end
            })

    return results


def main(audio_path: str, language: str = 'en',
         vad_json_path: Optional[str] = None,
         diarization_json_path: Optional[str] = None) -> Dict:
    """
    Main entry point for batched diarization transcription.

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
            - stats: Processing statistics including batch performance
    """
    if not vad_json_path:
        return {'error': 'VAD JSON required for diarization_batched method'}

    if not diarization_json_path:
        return {'error': 'Diarization JSON required for diarization_batched method'}

    logger.info(f"Starting batched diarization transcription")
    logger.info(f"  Batch size: {BATCH_SIZE}")

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

    # Create batches
    batches = create_batches(transcription_segments, BATCH_SIZE)
    logger.info(f"  Batches: {len(batches)} (avg {len(transcription_segments)/len(batches):.1f} segments/batch)")

    # Load Parakeet model once
    logger.info("Loading Parakeet v3 model...")
    start_load = time.time()
    model = load_parakeet_model()
    load_time = time.time() - start_load
    logger.info(f"  Model loaded in {load_time:.2f}s")

    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="transcribe_diar_batch_"))

    try:
        all_words = []
        segment_results = []

        start_time = time.time()

        for batch_idx, batch in enumerate(batches):
            logger.info(f"  [{batch_idx+1}/{len(batches)}] Processing batch with {len(batch)} segments")

            batch_results = transcribe_batch(model, audio, sr, batch, temp_dir)
            segment_results.extend(batch_results)

            for result in batch_results:
                if result['status'] == 'success':
                    all_words.extend(result['words'])

        processing_time = time.time() - start_time

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

        successful = sum(1 for r in segment_results if r['status'] == 'success')
        failed = sum(1 for r in segment_results if r['status'] == 'error')

        logger.info(f"Transcription complete:")
        logger.info(f"  Segments: {successful}/{len(transcription_segments)}")
        logger.info(f"  Words: {len(all_words)}")
        logger.info(f"  Processing time: {processing_time:.2f}s")
        logger.info(f"  Speed: {audio_duration/processing_time:.1f}x realtime")

        return {
            'segments': segments,
            'language': language,
            'method': METHOD_NAME,
            'stats': {
                'total_segments': len(transcription_segments),
                'successful': successful,
                'failed': failed,
                'total_words': len(all_words),
                'processing_time': processing_time,
                'model_load_time': load_time,
                'batch_size': BATCH_SIZE,
                'num_batches': len(batches),
                'realtime_factor': processing_time / audio_duration
            }
        }

    finally:
        # Clean up
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
