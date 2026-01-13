#!/usr/bin/env python3
"""
VAD Hybrid Two-Pass Transcription Method

Hybrid VAD two-pass transcription approach (PRODUCTION METHOD).

Strategy:
Pass 1: Efficient bulk transcription
    - Merge nearby VAD segments (eliminate silence/music gaps)
    - Let Parakeet focus on continuous speech chunks
    - Fast, handles 80-90% of content

Coverage Check:
    - Analyze EVERY VAD segment for transcription coverage
    - Identify segments with <80% coverage (early stopping victims)

Pass 2: Hyper-targeted precision re-transcription
    - Transcribe each under-covered VAD segment individually
    - Short segments avoid early stopping
    - Fills gaps, ensures completeness
"""

import os
import sys
import json
import time
import traceback
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import contextmanager

import torchaudio
import torch

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('vad_hybrid_2pass')

# Method metadata
METHOD_NAME = "vad_hybrid_2pass"
REQUIRES_VAD = True
REQUIRES_DIARIZATION = False

# Method-specific parameters (tune here)
MAX_GAP = 0.5                    # Maximum gap for merging VAD segments (seconds)
SPEECH_PADDING = 0.5             # Padding around speech segments (seconds)
MIN_COVERAGE_THRESHOLD = 0.8     # Minimum coverage ratio per VAD segment (0.0-1.0)
MAX_CHUNK_DURATION = 999999.0    # Maximum duration of merged chunks (effectively unlimited)


@contextmanager
def suppress_stderr():
    """Context manager to temporarily suppress stderr output from model loading"""
    with open(os.devnull, 'w') as devnull:
        stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = stderr


def load_parakeet_model(model_name="mlx-community/parakeet-tdt-0.6b-v3"):
    """Load Parakeet model once and return it for reuse"""
    logger.info(f"Loading Parakeet model: {model_name}...")
    with suppress_stderr():
        from parakeet_mlx import from_pretrained
        model = from_pretrained(model_name)
    logger.info("Parakeet model loaded successfully")
    return model


def _run_mlx_transcribe(audio_path: str, language: str = None, model=None) -> Dict:
    """
    Run Parakeet MLX v3 transcription.

    Args:
        audio_path: Path to audio file
        language: Language code (ISO 639-1)
        model: Optional pre-loaded Parakeet model

    Returns:
        Dict with transcription result
    """
    try:
        with suppress_stderr():
            parakeet_v3_languages = {
                'bg', 'hr', 'cs', 'da', 'nl', 'et', 'fi', 'fr', 'de',
                'el', 'hu', 'it', 'lv', 'lt', 'mt', 'pl', 'pt', 'ro', 'sk',
                'sl', 'es', 'sv', 'ru', 'uk', 'en'
            }

            use_parakeet_v3 = language and language in parakeet_v3_languages

            if use_parakeet_v3:
                try:
                    # Use provided model or load new one
                    if model is None:
                        model = load_parakeet_model()

                    # Run transcription
                    start_time = time.time()
                    result = model.transcribe(audio_path)
                    elapsed_time = time.time() - start_time

                    logger.debug(f"Parakeet transcription completed in {elapsed_time:.2f} seconds")

                    # Convert AlignedResult to standard format
                    text = ""
                    segments = []

                    if hasattr(result, 'text'):
                        text = result.text

                    # Extract sentence and word-level timestamps
                    if hasattr(result, 'sentences') and result.sentences:
                        for sentence in result.sentences:
                            words = []

                            # Merge tokens into words
                            if hasattr(sentence, 'tokens') and sentence.tokens:
                                current_word = None

                                for token in sentence.tokens:
                                    if not hasattr(token, 'text') or not hasattr(token, 'start') or not hasattr(token, 'end'):
                                        continue

                                    token_text = token.text

                                    # Check if this token starts a new word
                                    if token_text.startswith(' ') or current_word is None:
                                        # Save previous word if exists
                                        if current_word is not None:
                                            words.append(current_word)

                                        # Start new word
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

                                # Add final word
                                if current_word is not None:
                                    words.append(current_word)

                            # Create segment from sentence
                            segment = {
                                'text': sentence.text if hasattr(sentence, 'text') else '',
                                'start': float(sentence.start) if hasattr(sentence, 'start') else 0.0,
                                'end': float(sentence.end) if hasattr(sentence, 'end') else 0.0,
                                'words': words
                            }
                            segments.append(segment)
                    else:
                        # No sentence-level structure, create single segment
                        segments = [{
                            'text': text,
                            'start': 0.0,
                            'end': 0.0,
                            'words': []
                        }]

                    return {
                        'text': text,
                        'segments': segments,
                        'language': getattr(result, 'language', language),
                        'method': METHOD_NAME
                    }

                except Exception as parakeet_error:
                    logger.warning(f"Parakeet failed: {str(parakeet_error)}")
                    return {'error': str(parakeet_error)}

    except Exception as e:
        return {'error': str(e)}


def merge_nearby_segments(vad_segments: List[Dict], max_gap: float = MAX_GAP,
                         max_chunk_duration: float = MAX_CHUNK_DURATION) -> List[Dict]:
    """
    Merge VAD segments that are close together to create larger chunks.

    Args:
        vad_segments: List of VAD segments with 'start', 'end', 'duration' keys
        max_gap: Maximum gap between segments to merge (seconds)
        max_chunk_duration: Maximum duration of merged chunk (seconds)

    Returns:
        List of merged segments
    """
    if not vad_segments:
        return []

    import numpy as np

    # Extract start/end times as arrays
    starts = np.array([s['start'] for s in vad_segments])
    ends = np.array([s['end'] for s in vad_segments])

    # Calculate gaps between consecutive segments
    gaps = starts[1:] - ends[:-1]

    # Initialize: each segment is its own group initially
    merge_groups = [0]
    current_group = 0
    current_start = starts[0]

    for i, gap in enumerate(gaps):
        potential_duration = ends[i + 1] - current_start

        # Check if we should start a new group
        if gap >= max_gap or potential_duration > max_chunk_duration:
            current_group += 1
            current_start = starts[i + 1]

        merge_groups.append(current_group)

    # Build merged segments by group
    merge_groups = np.array(merge_groups)
    merged = []

    for group_id in range(current_group + 1):
        group_mask = merge_groups == group_id
        group_starts = starts[group_mask]
        group_ends = ends[group_mask]

        merged_seg = {
            'start': float(group_starts[0]),
            'end': float(group_ends[-1]),
            'duration': float(group_ends[-1] - group_starts[0])
        }
        merged.append(merged_seg)

    logger.info(f"Merged {len(vad_segments)} VAD segments into {len(merged)} chunks")
    return merged


def extract_vad_segments(audio_path: str, vad_segments: List[Dict],
                         output_dir: str, speech_padding: float = SPEECH_PADDING) -> List[Dict]:
    """
    Extract individual audio segments based on VAD boundaries, with padding.

    Args:
        audio_path: Path to original audio file
        vad_segments: List of VAD segments
        output_dir: Directory to save segment audio files
        speech_padding: Padding to add before/after each speech segment (seconds)

    Returns:
        List of segment info dicts
    """
    logger.info(f"Extracting {len(vad_segments)} VAD segments (padding: {speech_padding}s)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    wav = waveform.squeeze(0)
    audio_duration = len(wav) / sample_rate

    segment_infos = []

    for i, seg in enumerate(vad_segments):
        # Add padding to speech segment boundaries
        padded_start = max(0, seg['start'] - speech_padding)
        padded_end = min(audio_duration, seg['end'] + speech_padding)

        start_sample = int(padded_start * sample_rate)
        end_sample = int(padded_end * sample_rate)

        # Extract segment
        segment_audio = wav[start_sample:end_sample]

        # Save segment
        segment_path = output_dir / f"segment_{i:04d}.wav"
        torchaudio.save(str(segment_path), segment_audio.unsqueeze(0), sample_rate)

        segment_infos.append({
            'index': i,
            'audio_path': str(segment_path),
            'original_start': seg['start'],
            'original_end': seg['end'],
            'padded_start': padded_start,
            'padded_end': padded_end,
            'padding_start': seg['start'] - padded_start,
            'padding_end': padded_end - seg['end'],
            'duration': seg['end'] - seg['start']
        })

    logger.info(f"Extracted {len(segment_infos)} audio segments")
    return segment_infos


def transcribe_vad_segments(segment_infos: List[Dict], language: str = 'en', model=None) -> Dict:
    """
    Transcribe each VAD segment individually and combine results with remapped timestamps.

    Args:
        segment_infos: List of segment info dicts from extract_vad_segments()
        language: Language code for transcription
        model: Optional pre-loaded Parakeet model

    Returns:
        Combined transcription result dict
    """
    logger.info(f"Transcribing {len(segment_infos)} segments individually")

    all_segments = []
    total_words = 0

    for seg_info in segment_infos:
        logger.debug(f"Transcribing segment {seg_info['index']}: "
                    f"{seg_info['original_start']:.2f}-{seg_info['original_end']:.2f}s")

        # Transcribe this segment
        result = _run_mlx_transcribe(seg_info['audio_path'], language=language, model=model)

        if 'error' in result:
            logger.warning(f"Segment {seg_info['index']} transcription failed: {result['error']}")
            continue

        # Remap timestamps from segment audio to original timeline
        for seg in result.get('segments', []):
            seg_start_in_padded = seg.get('start', 0.0)
            seg_end_in_padded = seg.get('end', 0.0)

            padding_start = seg_info['padding_start']

            # Calculate position in original timeline
            original_seg_start = seg_info['original_start'] + (seg_start_in_padded - padding_start)
            original_seg_end = seg_info['original_start'] + (seg_end_in_padded - padding_start)

            remapped_seg = seg.copy()
            remapped_seg['start'] = original_seg_start
            remapped_seg['end'] = original_seg_end

            # Remap word timestamps
            if 'words' in seg:
                remapped_words = []
                for word in seg['words']:
                    word_start = word.get('start', 0.0)
                    word_end = word.get('end', 0.0)

                    original_word_start = seg_info['original_start'] + (word_start - padding_start)
                    original_word_end = seg_info['original_start'] + (word_end - padding_start)

                    remapped_word = word.copy()
                    remapped_word['start'] = original_word_start
                    remapped_word['end'] = original_word_end
                    remapped_words.append(remapped_word)

                remapped_seg['words'] = remapped_words
                total_words += len(remapped_words)

            all_segments.append(remapped_seg)

    return {
        'segments': all_segments,
        'language': language,
        'method': f"{METHOD_NAME}_segmented",
        'num_vad_segments': len(segment_infos),
        'total_segments': len(all_segments),
        'total_words': total_words
    }


def find_missing_vad_regions(transcription_result: Dict, vad_segments: List[Dict],
                             min_coverage_threshold: float = MIN_COVERAGE_THRESHOLD) -> List[Dict]:
    """
    Find VAD regions that lack adequate transcription coverage.

    Args:
        transcription_result: Transcription result dict with 'segments' key
        vad_segments: Original VAD segments to check coverage against
        min_coverage_threshold: Minimum coverage ratio required per VAD segment

    Returns:
        List of VAD segments needing re-transcription
    """
    import numpy as np

    transcribed_segments = transcription_result.get('segments', [])
    if not transcribed_segments:
        logger.warning("No transcribed segments found")
        return vad_segments

    missing_regions = []

    # Check each VAD segment individually
    for vad_seg in vad_segments:
        vad_start = vad_seg['start']
        vad_end = vad_seg['end']
        vad_duration = vad_seg['duration']

        # Find all transcription segments that overlap this VAD segment
        overlapping_trans = [
            s for s in transcribed_segments
            if not (s.get('end', 0) <= vad_start or s.get('start', 0) >= vad_end)
        ]

        # Calculate how much of the VAD segment is covered by transcription
        coverage = 0.0
        for trans_seg in overlapping_trans:
            overlap_start = max(trans_seg.get('start', 0), vad_start)
            overlap_end = min(trans_seg.get('end', 0), vad_end)
            overlap_duration = max(0, overlap_end - overlap_start)
            coverage += overlap_duration

        coverage_ratio = coverage / vad_duration if vad_duration > 0 else 0

        # If coverage is below threshold, mark for re-transcription
        if coverage_ratio < min_coverage_threshold:
            missing_regions.append({
                'start': vad_start,
                'end': vad_end,
                'duration': vad_duration,
                'coverage_ratio': coverage_ratio,
                'overlapping_segments': len(overlapping_trans)
            })

            logger.info(f"  VAD segment {vad_start:.2f}-{vad_end:.2f}s has only "
                       f"{coverage_ratio*100:.1f}% coverage")

    logger.info(f"Found {len(missing_regions)} VAD segments with inadequate coverage")
    return missing_regions


def main(
    audio_path: str,
    language: str = 'en',
    vad_json_path: Optional[str] = None,
    diarization_json_path: Optional[str] = None
) -> Dict:
    """
    Main entry point for hybrid VAD two-pass transcription.

    Args:
        audio_path: Path to audio file
        language: Language code (ISO 639-1)
        vad_json_path: Path to VAD JSON file (required)
        diarization_json_path: Ignored (method doesn't use diarization)

    Returns:
        Dict with transcription result
    """
    logger.info("="*80)
    logger.info("Hybrid VAD Two-Pass Transcription")
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

    temp_dir = Path(tempfile.mkdtemp(prefix="hybrid_vad_"))

    # Load Parakeet model once
    parakeet_model = load_parakeet_model()

    try:
        # ===== PASS 1: Efficient bulk transcription =====
        logger.info("\n[PASS 1] Bulk transcription with merged VAD chunks")
        logger.info("-" * 80)

        # Merge nearby segments
        merged_segments = merge_nearby_segments(vad_segments, max_gap=MAX_GAP,
                                               max_chunk_duration=MAX_CHUNK_DURATION)

        # Extract and transcribe merged chunks
        segments_dir = temp_dir / "pass1_segments"
        segment_infos = extract_vad_segments(audio_path, merged_segments,
                                            str(segments_dir), SPEECH_PADDING)

        pass1_result = transcribe_vad_segments(segment_infos, language=language, model=parakeet_model)

        if 'error' in pass1_result:
            logger.error(f"Pass 1 failed: {pass1_result['error']}")
            return pass1_result

        pass1_segments = pass1_result.get('segments', [])
        pass1_words = sum(len(seg.get('words', [])) for seg in pass1_segments)

        logger.info(f"✓ Pass 1 complete: {len(pass1_segments)} segments, {pass1_words} words")

        # ===== CHECK COVERAGE =====
        logger.info(f"\n[COVERAGE CHECK] Analyzing VAD segment coverage")
        logger.info("-" * 80)

        missing_regions = find_missing_vad_regions(pass1_result, vad_segments,
                                                   min_coverage_threshold=MIN_COVERAGE_THRESHOLD)

        if not missing_regions:
            logger.info("✓ All VAD segments adequately covered")
            return {
                'segments': pass1_segments,
                'language': language,
                'method': METHOD_NAME,
                'stats': {
                    'total_segments': len(pass1_segments),
                    'total_words': pass1_words,
                    'pass1_words': pass1_words,
                    'pass2_words': 0
                }
            }

        # ===== PASS 2: Precision re-transcription =====
        logger.info(f"\n[PASS 2] Hyper-targeted transcription of missing VAD segments")
        logger.info("-" * 80)

        pass2_segments_dir = temp_dir / "pass2_segments"
        pass2_segment_infos = extract_vad_segments(audio_path, missing_regions,
                                                   str(pass2_segments_dir), SPEECH_PADDING)

        pass2_result = transcribe_vad_segments(pass2_segment_infos, language=language, model=parakeet_model)

        if 'error' in pass2_result:
            logger.warning(f"Pass 2 failed, using Pass 1 result only")
            return pass1_result

        pass2_segments = pass2_result.get('segments', [])
        pass2_words = sum(len(seg.get('words', [])) for seg in pass2_segments)

        logger.info(f"✓ Pass 2 complete: {len(pass2_segments)} segments, {pass2_words} words")

        # ===== MERGE RESULTS =====
        logger.info("\n[MERGE] Combining Pass 1 and Pass 2 results")

        # Remove Pass 1 segments that overlap with missing VAD regions
        missing_ranges = [(r['start'], r['end']) for r in missing_regions]
        filtered_pass1 = []

        for seg in pass1_segments:
            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)

            overlaps_missing = False
            for missing_start, missing_end in missing_ranges:
                overlap_start = max(seg_start, missing_start)
                overlap_end = min(seg_end, missing_end)
                overlap_duration = max(0, overlap_end - overlap_start)

                seg_duration = seg_end - seg_start
                if seg_duration > 0 and overlap_duration / seg_duration > 0.5:
                    overlaps_missing = True
                    break

            if not overlaps_missing:
                filtered_pass1.append(seg)

        # Combine and sort
        all_segments = filtered_pass1 + pass2_segments
        all_segments.sort(key=lambda s: s.get('start', 0))

        # Filter out junk segments
        clean_segments = [
            seg for seg in all_segments
            if seg.get('text', '').strip() and not (seg.get('text', '').count('<unk>') > len(seg.get('text', '').split()) * 0.5)
        ]

        total_words = sum(len(seg.get('words', [])) for seg in clean_segments)

        logger.info(f"✓ Final result: {len(clean_segments)} segments, {total_words} words")

        return {
            'segments': clean_segments,
            'language': language,
            'method': METHOD_NAME,
            'stats': {
                'total_segments': len(clean_segments),
                'total_words': total_words,
                'pass1_words': pass1_words,
                'pass2_words': pass2_words
            }
        }

    finally:
        # Clean up temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
