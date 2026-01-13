#!/usr/bin/env python3
"""
Stage 13: Sentence-Level Emotion Detection
==========================================

Detects emotions in sentences using emotion2vec model.

Pipeline Order:
- Stage 12: Generate speaker turns + sentences → save to database
- Stage 13: Emotion detection on sentences → update sentences with emotion data
- Stage 14: Semantic segmentation → create segments with source_sentence_ids and emotion_summary

Key Responsibilities:
- Process each sentence's audio through emotion2vec
- Update sentence dicts with emotion, emotion_confidence, emotion_scores, arousal, valence, dominance
- Can be skipped (sentences will have emotion=None) and run as batch later

Input:
- Sentences from Stage 12 with text, start_time, end_time
- Audio file path or S3 content_id

Output:
- Same sentences list with emotion fields populated

Configuration:
- model_size: 'base' (~90M params) or 'large' (~300M params)
- min_duration: Minimum sentence duration to process (default: 0.5s)
"""

import os
# Force CPU usage - emotion2vec doesn't properly support MPS
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# Disable funasr progress bars
os.environ["FUNASR_DISABLE_PROGRESS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('stitch')

# Emotion labels from emotion2vec+
EMOTION_LABELS = [
    'angry', 'disgusted', 'fearful', 'happy',
    'neutral', 'other', 'sad', 'surprised', 'unknown'
]

# Dimensional mappings for arousal, valence, dominance
EMOTION_DIMENSIONS = {
    'angry': (0.85, 0.15, 0.75),
    'disgusted': (0.55, 0.20, 0.60),
    'fearful': (0.80, 0.10, 0.20),
    'happy': (0.70, 0.90, 0.70),
    'neutral': (0.30, 0.50, 0.50),
    'sad': (0.25, 0.20, 0.25),
    'surprised': (0.85, 0.55, 0.40),
    'other': (0.50, 0.50, 0.50),
    'unknown': (0.50, 0.50, 0.50),
    # Chinese/English keys from emotion2vec
    '生气/angry': (0.85, 0.15, 0.75),
    '厌恶/disgusted': (0.55, 0.20, 0.60),
    '恐惧/fearful': (0.80, 0.10, 0.20),
    '开心/happy': (0.70, 0.90, 0.70),
    '中立/neutral': (0.30, 0.50, 0.50),
    '难过/sad': (0.25, 0.20, 0.25),
    '吃惊/surprised': (0.85, 0.55, 0.40),
    '其他/other': (0.50, 0.50, 0.50),
    '<unk>': (0.50, 0.50, 0.50)
}


class SentenceEmotionDetector:
    """Detects emotions in sentences using wav2vec2 (MPS/CUDA) or emotion2vec (CPU)."""

    def __init__(self, model_size: str = 'large', backend: str = 'auto'):
        """Initialize the emotion detector.

        Args:
            model_size: 'base' or 'large' (for emotion2vec backend)
            backend: 'wav2vec2' (MPS/CUDA), 'emotion2vec' (CPU), or 'auto' (prefer wav2vec2 if MPS available)
        """
        self.model_size = model_size
        self.backend = backend
        self.model = None
        self.processor = None
        self.device = None
        self.model_loaded = False
        self.min_duration = 0.5  # Absolute minimum - skip if shorter
        self.group_threshold = 2.0  # Group with adjacent if shorter than this
        self.temp_dir = None

    def _load_model(self) -> bool:
        """Lazily load the emotion model."""
        if self.model_loaded:
            return True

        # Determine backend
        actual_backend = self.backend
        if actual_backend == 'auto':
            import torch
            if torch.backends.mps.is_available():
                actual_backend = 'wav2vec2'
                logger.info("MPS available, using wav2vec2 backend")
            elif torch.cuda.is_available():
                actual_backend = 'wav2vec2'
                logger.info("CUDA available, using wav2vec2 backend")
            else:
                actual_backend = 'emotion2vec'
                logger.info("No GPU available, using emotion2vec (CPU) backend")

        if actual_backend == 'wav2vec2':
            return self._load_wav2vec2_model()
        else:
            return self._load_emotion2vec_model()

    def _load_wav2vec2_model(self) -> bool:
        """Load wav2vec2 model for MPS/CUDA."""
        try:
            import torch
            from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

            # Select device
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS (Apple Silicon) for emotion detection")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA for emotion detection")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU for emotion detection")

            model_name = "superb/wav2vec2-base-superb-er"
            logger.info(f"Loading wav2vec2 emotion model: {model_name}")

            # Use FeatureExtractor (not Processor) - this model doesn't have a tokenizer
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            # Map model's short labels to full names: neu->neutral, hap->happy, ang->angry, sad->sad
            self.wav2vec2_label_map = {'neu': 'neutral', 'hap': 'happy', 'ang': 'angry', 'sad': 'sad'}
            self.wav2vec2_id2label = self.model.config.id2label  # {0: 'neu', 1: 'hap', 2: 'ang', 3: 'sad'}

            self.backend = 'wav2vec2'
            self.model_loaded = True
            logger.info(f"wav2vec2 emotion model loaded on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load wav2vec2 model: {e}")
            logger.info("Falling back to emotion2vec")
            return self._load_emotion2vec_model()

    def _load_emotion2vec_model(self) -> bool:
        """Load emotion2vec model (CPU only)."""
        try:
            import warnings
            import sys
            import os
            import logging as std_logging

            # Suppress all funasr/emotion2vec warnings
            warnings.filterwarnings('ignore', message='.*miss key in ckpt.*')
            warnings.filterwarnings('ignore', message='.*trust_remote_code.*')
            warnings.filterwarnings('ignore')

            # Suppress root logger warnings
            std_logging.getLogger().setLevel(std_logging.ERROR)
            std_logging.getLogger('funasr').setLevel(std_logging.ERROR)
            std_logging.getLogger('modelscope').setLevel(std_logging.ERROR)

            model_name = f"iic/emotion2vec_plus_{self.model_size}"
            logger.info(f"Loading emotion2vec model: {model_name}")

            # Suppress stdout/stderr during model loading
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

            try:
                from funasr import AutoModel
                self.model = AutoModel(
                    model=model_name,
                    hub="hf",
                    device="cpu",
                    disable_update=True,
                    disable_pbar=True
                )
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            self.backend = 'emotion2vec'
            self.model_loaded = True
            logger.info(f"emotion2vec model ({self.model_size}) loaded successfully")
            return True

        except ImportError as e:
            logger.error(f"Failed to import funasr. Install with: pip install funasr")
            logger.error(f"Error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load emotion2vec model: {e}")
            return False

    def _compute_dimensional_scores(self, emotion_scores: Dict[str, float]) -> Tuple[float, float, float]:
        """Compute arousal, valence, dominance from emotion probabilities."""
        arousal = 0.0
        valence = 0.0
        dominance = 0.0

        for emotion, prob in emotion_scores.items():
            if emotion in EMOTION_DIMENSIONS:
                dims = EMOTION_DIMENSIONS[emotion]
                arousal += prob * dims[0]
                valence += prob * dims[1]
                dominance += prob * dims[2]

        return (
            max(0.0, min(1.0, arousal)),
            max(0.0, min(1.0, valence)),
            max(0.0, min(1.0, dominance))
        )

    def _process_audio_window(self, audio_segment: np.ndarray, sample_rate: int,
                              temp_path: Path) -> Optional[np.ndarray]:
        """Process a single audio window, returning raw logits.

        Args:
            audio_segment: Audio data as numpy array
            sample_rate: Sample rate
            temp_path: Path for temporary wav file (only used for emotion2vec)

        Returns:
            Raw logits array or None if failed
        """
        if self.backend == 'wav2vec2':
            return self._process_audio_window_wav2vec2(audio_segment, sample_rate)
        else:
            return self._process_audio_window_emotion2vec(audio_segment, sample_rate, temp_path)

    def _process_audio_window_wav2vec2(self, audio_segment: np.ndarray, sample_rate: int) -> Optional[np.ndarray]:
        """Process audio through wav2vec2 model."""
        try:
            import torch

            # Process audio
            inputs = self.processor(
                audio_segment,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                logits = self.model(**inputs).logits

            return logits.cpu().numpy()[0]

        except Exception as e:
            logger.warning(f"Failed to process audio window (wav2vec2): {e}")
            return None

    def _process_audio_window_emotion2vec(self, audio_segment: np.ndarray, sample_rate: int,
                                          temp_path: Path) -> Optional[np.ndarray]:
        """Process audio through emotion2vec model."""
        try:
            import soundfile as sf

            # Save to temp file
            sf.write(str(temp_path), audio_segment, sample_rate)

            # Run emotion2vec (suppress stdout/stderr output from funasr)
            import sys
            import os
            from io import StringIO
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = StringIO()
            sys.stderr = open(os.devnull, 'w')
            try:
                result = self.model.generate(
                    str(temp_path),
                    granularity="utterance",
                    extract_embedding=False,
                    disable_pbar=True
                )
            finally:
                sys.stdout = old_stdout
                sys.stderr.close()
                sys.stderr = old_stderr

            if not result or len(result) == 0:
                return None

            result_data = result[0]
            scores = result_data.get('scores', [])

            if not scores:
                return None

            return np.array(scores)

        except Exception as e:
            logger.warning(f"Failed to process audio window (emotion2vec): {e}")
            return None
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _process_sentence_audio(self, audio_segment: np.ndarray, sample_rate: int,
                                temp_path: Path, sentence_duration: float) -> Optional[Dict]:
        """Process a sentence's audio through the emotion model.

        For sentences longer than max_window_size (15s), splits into windows
        and uses logit averaging before softmax.

        Args:
            audio_segment: Audio data as numpy array
            sample_rate: Sample rate
            temp_path: Path for temporary wav file
            sentence_duration: Duration of sentence in seconds

        Returns:
            Dict with emotion data or None if failed
        """
        max_window_size = 15.0  # Maximum window size in seconds
        window_size = 8.0  # Optimal window size

        try:
            # For short sentences, process directly
            if sentence_duration <= max_window_size:
                logits = self._process_audio_window(audio_segment, sample_rate, temp_path)
                if logits is None:
                    return None
                all_logits = [logits]
            else:
                # Long sentence: split into windows and collect logits
                logger.debug(f"Long sentence ({sentence_duration:.1f}s) - using windowed processing")
                all_logits = []
                window_samples = int(window_size * sample_rate)
                step_samples = window_samples  # Non-overlapping windows

                pos = 0
                window_idx = 0
                while pos < len(audio_segment):
                    end_pos = min(pos + window_samples, len(audio_segment))

                    # Only process if window is at least min_duration
                    window_duration = (end_pos - pos) / sample_rate
                    if window_duration >= self.min_duration:
                        window_audio = audio_segment[pos:end_pos]
                        window_path = temp_path.parent / f"{temp_path.stem}_w{window_idx}.wav"
                        logits = self._process_audio_window(window_audio, sample_rate, window_path)
                        if logits is not None:
                            all_logits.append(logits)
                        window_idx += 1

                    pos += step_samples

                if not all_logits:
                    return None

                logger.debug(f"Processed {len(all_logits)} windows for long sentence")

            # Average logits across windows, then apply softmax
            logits_stack = np.stack(all_logits)
            mean_logits = np.mean(logits_stack, axis=0)

            # Softmax
            exp_logits = np.exp(mean_logits - np.max(mean_logits))
            probs = exp_logits / exp_logits.sum()

            # Get labels based on backend
            if self.backend == 'wav2vec2':
                # wav2vec2 outputs: 0=neu, 1=hap, 2=ang, 3=sad
                # Map to full names: neutral, happy, angry, sad
                labels = ['neutral', 'happy', 'angry', 'sad']
            else:
                labels = EMOTION_LABELS  # emotion2vec's 9 labels

            # Find primary emotion
            primary_idx = np.argmax(probs)
            primary_emotion = labels[primary_idx] if primary_idx < len(labels) else 'unknown'

            # Create emotion scores dict
            emotion_scores = {}
            for i, prob in enumerate(probs):
                if i < len(labels):
                    label = labels[i]
                    # Normalize label (remove Chinese prefix if present for emotion2vec)
                    normalized_label = label.split('/')[-1] if '/' in label else label
                    emotion_scores[normalized_label] = float(prob)

            # Compute dimensional scores
            arousal, valence, dominance = self._compute_dimensional_scores(emotion_scores)

            return {
                'emotion': primary_emotion,
                'emotion_confidence': float(probs[primary_idx]),
                'emotion_scores': emotion_scores,
                'arousal': arousal,
                'valence': valence,
                'dominance': dominance,
                'windows_processed': len(all_logits),
                'backend': self.backend
            }

        except Exception as e:
            logger.warning(f"Failed to process sentence audio: {e}")
            return None

    def _build_sentence_groups(self, sentences: List[Dict]) -> List[List[int]]:
        """Build groups of sentence indices for processing.

        Short sentences (<2s) are grouped with adjacent same-speaker sentences
        to provide enough audio context for emotion detection.

        Args:
            sentences: List of sentence dicts

        Returns:
            List of groups, where each group is a list of sentence indices
        """
        if not sentences:
            return []

        groups = []
        processed_indices = set()

        for i, sentence in enumerate(sentences):
            if i in processed_indices:
                continue

            duration = sentence['end_time'] - sentence['start_time']
            speaker = sentence.get('speaker_label') or sentence.get('speaker_id')

            # If sentence is long enough, process individually
            if duration >= self.group_threshold:
                groups.append([i])
                processed_indices.add(i)
                continue

            # Short sentence - try to group with adjacent same-speaker sentences
            group = [i]
            group_duration = duration
            processed_indices.add(i)

            # Look forward for adjacent same-speaker sentences
            j = i + 1
            while j < len(sentences) and group_duration < self.group_threshold:
                next_sentence = sentences[j]
                next_speaker = next_sentence.get('speaker_label') or next_sentence.get('speaker_id')

                # Must be same speaker
                if next_speaker != speaker:
                    break

                # Check if temporally adjacent (within 1 second gap)
                gap = next_sentence['start_time'] - sentences[group[-1]]['end_time']
                if gap > 1.0:
                    break

                next_duration = next_sentence['end_time'] - next_sentence['start_time']
                group.append(j)
                group_duration += next_duration
                processed_indices.add(j)
                j += 1

            groups.append(group)

        return groups

    def process_sentences(self, sentences: List[Dict], audio_data: np.ndarray,
                         sample_rate: int, content_id: str) -> List[Dict]:
        """Process all sentences and add emotion data.

        Short sentences (<2s) are grouped with adjacent same-speaker sentences
        to provide enough audio context for reliable emotion detection.

        Args:
            sentences: List of sentence dicts from stage12
            audio_data: Full audio as numpy array
            sample_rate: Audio sample rate
            content_id: Content ID for logging

        Returns:
            Same sentences list with emotion fields populated
        """
        if not self._load_model():
            logger.error(f"[{content_id}] Cannot process emotions - model failed to load")
            return sentences

        # Create temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix="emotion_"))

        try:
            # Build sentence groups (short sentences grouped with adjacent same-speaker)
            groups = self._build_sentence_groups(sentences)

            processed = 0
            skipped = 0
            grouped = 0

            for group_idx, group in enumerate(groups):
                # Get group boundaries
                first_sentence = sentences[group[0]]
                last_sentence = sentences[group[-1]]
                group_start = first_sentence['start_time']
                group_end = last_sentence['end_time']
                group_duration = group_end - group_start

                # Skip if still too short after grouping
                if group_duration < self.min_duration:
                    skipped += len(group)
                    continue

                # Extract audio for the group
                start_sample = int(group_start * sample_rate)
                end_sample = int(group_end * sample_rate)
                start_sample = max(0, start_sample)
                end_sample = min(len(audio_data), end_sample)

                if end_sample <= start_sample:
                    skipped += len(group)
                    continue

                audio_segment = audio_data[start_sample:end_sample]

                # Process through emotion2vec
                temp_path = self.temp_dir / f"group_{group_idx}.wav"
                emotion_result = self._process_sentence_audio(audio_segment, sample_rate, temp_path, group_duration)

                if emotion_result:
                    # Apply emotion to all sentences in the group
                    for sent_idx in group:
                        sentences[sent_idx].update(emotion_result)
                    processed += len(group)
                    if len(group) > 1:
                        grouped += len(group)
                else:
                    skipped += len(group)

                # Progress logging
                if (group_idx + 1) % 50 == 0:
                    total_processed = sum(len(g) for g in groups[:group_idx + 1])
                    logger.info(f"[{content_id}] Processed {total_processed}/{len(sentences)} sentences ({group_idx + 1}/{len(groups)} groups)")

            logger.info(f"[{content_id}] Emotion detection complete: {processed} processed ({grouped} via grouping), {skipped} skipped, {len(groups)} groups from {len(sentences)} sentences")
            return sentences

        finally:
            # Cleanup
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None


def stage13_emotion(content_id: str, sentences: List[Dict],
                   audio_path: Optional[str] = None,
                   s3_storage=None,
                   model_size: str = 'large',
                   test_mode: bool = False) -> Dict[str, Any]:
    """
    Execute Stage 13: Sentence-Level Emotion Detection.

    Processes each sentence through emotion2vec and updates emotion fields.

    Args:
        content_id: Content ID
        sentences: List of sentence dictionaries from Stage 12
        audio_path: Local path to audio file (if available)
        s3_storage: S3Storage instance for downloading audio (if audio_path not provided)
        model_size: 'base' or 'large'
        test_mode: If True, skip heavy processing

    Returns:
        Dictionary with status and updated sentences
    """
    start_time = time.time()

    logger.info(f"[{content_id}] Starting Stage 13: Emotion Detection")

    result = {
        'status': 'pending',
        'content_id': content_id,
        'stage': 'stage13_emotion',
        'data': {
            'sentences': sentences
        },
        'stats': {},
        'error': None
    }

    try:
        if not sentences:
            logger.warning(f"[{content_id}] No sentences to process for emotion detection")
            result['status'] = 'success'
            result['stats'] = {'sentences_processed': 0, 'duration': 0}
            return result

        # Get audio
        audio_data = None
        sample_rate = 16000
        temp_audio_path = None

        if audio_path and Path(audio_path).exists():
            # Use local audio
            import librosa
            audio_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
            logger.info(f"[{content_id}] Loaded local audio: {len(audio_data)/sample_rate:.1f}s")

        elif s3_storage:
            # Download from S3
            temp_dir = Path(tempfile.mkdtemp(prefix="emotion_audio_"))
            temp_audio_path = temp_dir / "audio.wav"

            try:
                if s3_storage.download_audio_flexible(content_id, str(temp_audio_path)):
                    import librosa
                    audio_data, sample_rate = librosa.load(str(temp_audio_path), sr=16000, mono=True)
                    logger.info(f"[{content_id}] Downloaded audio from S3: {len(audio_data)/sample_rate:.1f}s")
                else:
                    raise ValueError(f"Failed to download audio for {content_id}")
            finally:
                if temp_audio_path and temp_audio_path.exists():
                    shutil.rmtree(temp_dir)

        else:
            logger.warning(f"[{content_id}] No audio available - skipping emotion detection")
            result['status'] = 'success'
            result['stats'] = {'sentences_processed': 0, 'skipped': 'no_audio', 'duration': time.time() - start_time}
            return result

        # Process sentences
        detector = SentenceEmotionDetector(model_size=model_size)
        updated_sentences = detector.process_sentences(sentences, audio_data, sample_rate, content_id)

        # Count results
        with_emotion = sum(1 for s in updated_sentences if s.get('emotion'))

        result['data']['sentences'] = updated_sentences
        result['status'] = 'success'
        result['stats'] = {
            'duration': time.time() - start_time,
            'sentences_total': len(sentences),
            'sentences_with_emotion': with_emotion,
            'model_size': model_size
        }

        logger.info(f"[{content_id}] Stage 13 completed: {with_emotion}/{len(sentences)} sentences with emotion")
        return result

    except Exception as e:
        logger.error(f"[{content_id}] Stage 13 failed: {str(e)}", exc_info=True)
        result.update({
            'status': 'error',
            'error': str(e),
            'stats': {'duration': time.time() - start_time}
        })
        return result


def update_sentence_emotions(sentences: List[Dict], content_id: str, test_mode: bool = False) -> int:
    """
    Update sentence emotion fields in the database after emotion detection.

    This is called after stage13_emotion completes to persist the emotion data
    that was computed for each sentence.

    Args:
        sentences: List of sentence dicts with emotion fields populated
        content_id: Platform content ID (e.g., YouTube video ID)
        test_mode: If True, only log what would be updated

    Returns:
        Number of sentences updated
    """
    if not sentences:
        return 0

    # Count how many have emotion data
    with_emotion = sum(1 for s in sentences if s.get('emotion'))
    if with_emotion == 0:
        logger.info(f"[{content_id}] No sentences with emotion data to update")
        return 0

    if test_mode:
        logger.info(f"[{content_id}] TEST MODE: Would update {with_emotion} sentences with emotion data")
        return 0

    try:
        from src.database.models import Sentence, Content
        from src.database.session import get_session

        updated_count = 0

        with get_session() as session:
            # Get the content record to get internal ID
            content = session.query(Content).filter_by(content_id=content_id).first()
            if not content:
                logger.error(f"[{content_id}] Content not found in database")
                return 0

            # Update each sentence with emotion data
            for sent in sentences:
                if not sent.get('emotion'):
                    continue

                # Find the sentence by content_id and sentence_index
                sentence_record = session.query(Sentence).filter_by(
                    content_id=content.id,
                    sentence_index=sent['sentence_index']
                ).first()

                if sentence_record:
                    sentence_record.emotion = sent.get('emotion')
                    sentence_record.emotion_confidence = sent.get('emotion_confidence')
                    sentence_record.emotion_scores = sent.get('emotion_scores')
                    sentence_record.arousal = sent.get('arousal')
                    sentence_record.valence = sent.get('valence')
                    sentence_record.dominance = sent.get('dominance')
                    updated_count += 1

            session.commit()
            logger.info(f"[{content_id}] Updated {updated_count} sentences with emotion data")

        return updated_count

    except Exception as e:
        logger.error(f"[{content_id}] Failed to update sentence emotions: {e}", exc_info=True)
        return 0
