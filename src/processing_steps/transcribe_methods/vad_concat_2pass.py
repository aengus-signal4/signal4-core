#!/usr/bin/env python3
"""
VAD Concatenation Two-Pass Transcription Method

Optimized VAD-Assisted 2-Pass Transcription Pipeline:
- Pass 1: Remove silence/music, concatenate speech, transcribe in bulk
- Pass 2: Check coverage, re-transcribe under-covered segments

Key improvements:
- Configurable silence gap removal (>1s with 0.25s padding)
- Proper timestamp remapping from concatenated to original timeline
- Coverage validation ensuring >80% VAD segment coverage
"""

import os
import sys
import json
import time
import traceback
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager

import numpy as np
import torchaudio
import torch

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('vad_concat_2pass')

# Method metadata
METHOD_NAME = "vad_concat_2pass"
REQUIRES_VAD = True
REQUIRES_DIARIZATION = False

# Method-specific parameters (tune here)
MIN_GAP_DURATION = 1.0        # Minimum gap duration to remove (seconds)
PADDING = 0.25                # Padding to keep between segments (seconds)
MAX_CHUNK_DURATION = 600.0    # Maximum duration per chunk (seconds)
COVERAGE_THRESHOLD = 0.8      # Minimum coverage ratio required (0.0-1.0)


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


def load_parakeet_model(model_name="mlx-community/parakeet-tdt-0.6b-v3"):
    """Load Parakeet model"""
    logger.info(f"Loading Parakeet model: {model_name}...")
    with suppress_stderr():
        from parakeet_mlx import from_pretrained
        model = from_pretrained(model_name)
    logger.info("Parakeet model loaded")
    return model


def _run_mlx_transcribe(audio_path: str, language: str = None, model=None) -> Dict:
    """Run Parakeet MLX v3 transcription"""
    try:
        with suppress_stderr():
            parakeet_v3_languages = {
                'bg', 'hr', 'cs', 'da', 'nl', 'et', 'fi', 'fr', 'de',
                'el', 'hu', 'it', 'lv', 'lt', 'mt', 'pl', 'pt', 'ro', 'sk',
                'sl', 'es', 'sv', 'ru', 'uk', 'en'
            }

            use_parakeet = language and language in parakeet_v3_languages

            if use_parakeet:
                try:
                    if model is None:
                        model = load_parakeet_model()

                    start_time = time.time()
                    result = model.transcribe(audio_path)
                    elapsed_time = time.time() - start_time

                    logger.info(f"Parakeet transcription completed in {elapsed_time:.2f}s")

                    # Convert to standard format
                    text = result.text if hasattr(result, 'text') else ""
                    segments = []

                    if hasattr(result, 'sentences') and result.sentences:
                        for sentence in result.sentences:
                            words = []
                            if hasattr(sentence, 'tokens') and sentence.tokens:
                                current_word = None
                                for token in sentence.tokens:
                                    if not all(hasattr(token, attr) for attr in ['text', 'start', 'end']):
                                        continue
                                    token_text = token.text
                                    if token_text.startswith(' ') or current_word is None:
                                        if current_word is not None:
                                            words.append(current_word)
                                        current_word = {
                                            'word': token_text.strip(),
                                            'start': float(token.start),
                                            'end': float(token.end)
                                        }
                                    else:
                                        current_word['word'] += token_text
                                        current_word['end'] = float(token.end)
                                if current_word is not None:
                                    words.append(current_word)

                            segment = {
                                'text': sentence.text if hasattr(sentence, 'text') else '',
                                'start': float(sentence.start) if hasattr(sentence, 'start') else 0.0,
                                'end': float(sentence.end) if hasattr(sentence, 'end') else 0.0,
                                'words': words
                            }
                            segments.append(segment)

                    return {
                        'text': text,
                        'segments': segments,
                        'language': getattr(result, 'language', language),
                        'method': METHOD_NAME
                    }

                except Exception as e:
                    logger.warning(f"Parakeet failed: {e}")
                    return {'error': str(e)}

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {'error': str(e)}


def remove_silence_and_concatenate(
    audio_path: str,
    vad_segments: List[Dict],
    min_gap_duration: float = MIN_GAP_DURATION,
    padding: float = PADDING,
    max_chunk_duration: float = MAX_CHUNK_DURATION,
    output_dir: Optional[str] = None
) -> Tuple[List[str], List[List[Dict]]]:
    """
    Remove silence gaps and concatenate speech segments.

    Returns:
        Tuple of (chunk_paths, chunk_mappings)
    """
    logger.info(f"Processing {len(vad_segments)} VAD segments")
    logger.info(f"Removing gaps >{min_gap_duration}s, keeping {padding}s padding")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="vad_concat_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    wav = waveform.squeeze(0)
    total_duration = len(wav) / sample_rate

    # Build concatenated segments
    concatenated_segments = []
    timestamp_mappings = []
    current_concat_time = 0.0

    for i, seg in enumerate(vad_segments):
        seg_start = seg['start']
        seg_end = seg['end']
        seg_duration = seg_end - seg_start

        # Check gap from previous segment
        if i > 0:
            prev_end = vad_segments[i-1]['end']
            gap_duration = seg_start - prev_end

            if gap_duration > min_gap_duration:
                # Add minimal padding
                gap_samples = int(padding * sample_rate)
                gap_start_sample = int((seg_start - padding) * sample_rate)
                gap_end_sample = int(seg_start * sample_rate)

                if gap_start_sample >= 0 and gap_end_sample <= len(wav):
                    gap_audio = wav[gap_start_sample:gap_end_sample]
                    concatenated_segments.append(gap_audio)

                    timestamp_mappings.append({
                        'concat_start': current_concat_time,
                        'concat_end': current_concat_time + padding,
                        'original_start': seg_start - padding,
                        'original_end': seg_start,
                        'vad_index': i,
                        'is_padding': True
                    })

                    current_concat_time += padding

        # Extract VAD segment audio
        start_sample = int(seg_start * sample_rate)
        end_sample = int(seg_end * sample_rate)
        segment_audio = wav[start_sample:end_sample]

        timestamp_mappings.append({
            'concat_start': current_concat_time,
            'concat_end': current_concat_time + seg_duration,
            'original_start': seg_start,
            'original_end': seg_end,
            'vad_index': i
        })

        concatenated_segments.append(segment_audio)
        current_concat_time += seg_duration

    # Concatenate all segments
    if not concatenated_segments:
        return [], []

    concatenated_audio = torch.cat(concatenated_segments, dim=0)
    concat_duration = len(concatenated_audio) / sample_rate

    logger.info(f"Concatenated: {total_duration:.1f}s -> {concat_duration:.1f}s")

    # Save concatenated audio
    chunk_path = output_dir / "chunk_0.wav"
    torchaudio.save(str(chunk_path), concatenated_audio.unsqueeze(0), sample_rate)

    return [str(chunk_path)], [timestamp_mappings]


def remap_timestamps_to_original(
    segments: List[Dict],
    timestamp_mappings: List[Dict],
    chunk_offset: float = 0.0
) -> List[Dict]:
    """Remap timestamps from concatenated audio back to original timeline"""
    remapped_segments = []

    for seg in segments:
        seg_start = seg.get('start', 0) + chunk_offset
        seg_end = seg.get('end', 0) + chunk_offset

        # Find which VAD segment this transcription belongs to
        original_start = None
        original_end = None

        for mapping in timestamp_mappings:
            if mapping['concat_start'] <= seg_start <= mapping['concat_end']:
                offset_in_concat = seg_start - mapping['concat_start']
                original_start = mapping['original_start'] + offset_in_concat

            if mapping['concat_start'] <= seg_end <= mapping['concat_end']:
                offset_in_concat = seg_end - mapping['concat_start']
                original_end = mapping['original_start'] + offset_in_concat

        # Fallback if not found
        if original_start is None:
            original_start = seg_start
        if original_end is None:
            original_end = seg_end

        remapped_seg = seg.copy()
        remapped_seg['start'] = original_start
        remapped_seg['end'] = original_end

        # Remap word timestamps
        if 'words' in seg:
            remapped_words = []
            for word in seg['words']:
                word_start = word.get('start', 0) + chunk_offset
                word_end = word.get('end', 0) + chunk_offset

                original_word_start = None
                original_word_end = None

                for mapping in timestamp_mappings:
                    if mapping['concat_start'] <= word_start <= mapping['concat_end']:
                        offset_in_concat = word_start - mapping['concat_start']
                        original_word_start = mapping['original_start'] + offset_in_concat
                    if mapping['concat_start'] <= word_end <= mapping['concat_end']:
                        offset_in_concat = word_end - mapping['concat_start']
                        original_word_end = mapping['original_start'] + offset_in_concat

                if original_word_start is None:
                    original_word_start = original_start
                if original_word_end is None:
                    original_word_end = original_end

                remapped_word = word.copy()
                remapped_word['start'] = original_word_start
                remapped_word['end'] = original_word_end
                remapped_words.append(remapped_word)

            remapped_seg['words'] = remapped_words

        remapped_segments.append(remapped_seg)

    return remapped_segments


def check_vad_coverage(
    transcription_segments: List[Dict],
    vad_segments: List[Dict],
    threshold: float = COVERAGE_THRESHOLD
) -> List[Dict]:
    """Check if each VAD segment has adequate transcription coverage"""
    under_covered = []

    for vad_seg in vad_segments:
        vad_start = vad_seg['start']
        vad_end = vad_seg['end']
        vad_duration = vad_end - vad_start

        coverage = 0.0
        for trans_seg in transcription_segments:
            trans_start = trans_seg.get('start', 0)
            trans_end = trans_seg.get('end', 0)

            overlap_start = max(vad_start, trans_start)
            overlap_end = min(vad_end, trans_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            coverage += overlap_duration

        coverage_ratio = coverage / vad_duration if vad_duration > 0 else 0

        if coverage_ratio < threshold:
            under_covered.append({
                'start': vad_start,
                'end': vad_end,
                'duration': vad_duration,
                'coverage_ratio': coverage_ratio
            })

    logger.info(f"Found {len(under_covered)} VAD segments with <{threshold*100:.0f}% coverage")
    return under_covered


def transcribe_individual_segments(
    audio_path: str,
    segments: List[Dict],
    language: str = 'en',
    padding: float = 0.5,
    model=None
) -> List[Dict]:
    """Transcribe individual VAD segments"""
    logger.info(f"Transcribing {len(segments)} individual segments")

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    wav = waveform.squeeze(0)
    audio_duration = len(wav) / sample_rate

    all_segments = []
    temp_dir = Path(tempfile.mkdtemp(prefix="vad_individual_"))

    try:
        for i, seg in enumerate(segments):
            padded_start = max(0, seg['start'] - padding)
            padded_end = min(audio_duration, seg['end'] + padding)

            start_sample = int(padded_start * sample_rate)
            end_sample = int(padded_end * sample_rate)
            segment_audio = wav[start_sample:end_sample]

            segment_path = temp_dir / f"segment_{i}.wav"
            torchaudio.save(str(segment_path), segment_audio.unsqueeze(0), sample_rate)

            result = _run_mlx_transcribe(str(segment_path), language, model)

            if 'error' not in result:
                padding_offset = seg['start'] - padded_start
                for trans_seg in result.get('segments', []):
                    trans_seg['start'] = seg['start'] + (trans_seg.get('start', 0) - padding_offset)
                    trans_seg['end'] = seg['start'] + (trans_seg.get('end', 0) - padding_offset)

                    if 'words' in trans_seg:
                        for word in trans_seg['words']:
                            word['start'] = seg['start'] + (word.get('start', 0) - padding_offset)
                            word['end'] = seg['start'] + (word.get('end', 0) - padding_offset)

                    all_segments.append(trans_seg)

        return all_segments
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def main(
    audio_path: str,
    language: str = 'en',
    vad_json_path: Optional[str] = None,
    diarization_json_path: Optional[str] = None
) -> Dict:
    """
    Main entry point for VAD concatenation two-pass transcription.

    Returns:
        Dict with transcription result
    """
    logger.info("="*80)
    logger.info("VAD Concatenation Two-Pass Transcription")
    logger.info("="*80)

    if not vad_json_path:
        return {'error': 'VAD JSON path required for this method'}

    # Load VAD segments
    with open(vad_json_path) as f:
        vad_data = json.load(f)
    vad_segments = vad_data.get('segments', [])

    logger.info(f"Loaded {len(vad_segments)} VAD segments")

    if not vad_segments:
        return {
            'segments': [],
            'language': language,
            'method': METHOD_NAME,
            'stats': {'total_words': 0}
        }

    temp_dir = Path(tempfile.mkdtemp(prefix="vad_concat_2pass_"))
    model = load_parakeet_model()

    try:
        # PASS 1: Concatenated transcription
        logger.info("\n[PASS 1] Concatenated transcription")
        chunk_paths, chunk_mappings = remove_silence_and_concatenate(
            audio_path, vad_segments, MIN_GAP_DURATION, PADDING,
            MAX_CHUNK_DURATION, str(temp_dir / "concatenated")
        )

        pass1_segments = []
        for chunk_path, mappings in zip(chunk_paths, chunk_mappings):
            result = _run_mlx_transcribe(chunk_path, language, model)
            if 'error' not in result:
                remapped = remap_timestamps_to_original(result.get('segments', []), mappings, 0)
                pass1_segments.extend(remapped)

        pass1_words = sum(len(seg.get('words', [])) for seg in pass1_segments)
        logger.info(f"✓ Pass 1: {len(pass1_segments)} segments, {pass1_words} words")

        # Check coverage
        under_covered = check_vad_coverage(pass1_segments, vad_segments, COVERAGE_THRESHOLD)

        if not under_covered:
            return {
                'segments': pass1_segments,
                'language': language,
                'method': METHOD_NAME,
                'stats': {
                    'total_words': pass1_words,
                    'pass1_words': pass1_words,
                    'pass2_words': 0
                }
            }

        # PASS 2: Re-transcribe under-covered segments
        logger.info(f"\n[PASS 2] Re-transcribing {len(under_covered)} under-covered segments")
        pass2_segments = transcribe_individual_segments(audio_path, under_covered, language, 0.5, model)
        pass2_words = sum(len(seg.get('words', [])) for seg in pass2_segments)

        # Merge results
        all_segments = pass1_segments + pass2_segments
        all_segments.sort(key=lambda s: s.get('start', 0))
        total_words = sum(len(seg.get('words', [])) for seg in all_segments)

        return {
            'segments': all_segments,
            'language': language,
            'method': METHOD_NAME,
            'stats': {
                'total_words': total_words,
                'pass1_words': pass1_words,
                'pass2_words': pass2_words
            }
        }

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
