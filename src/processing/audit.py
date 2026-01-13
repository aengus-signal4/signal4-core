#!/usr/bin/env python3
"""
Content Audit Script
====================

Audits a specific content item by:
1. Evaluating current state vs S3 reality
2. Updating database flags to match actual files
3. Creating any missing tasks needed for processing
4. Providing detailed report of findings and actions

Usage:
    python audit_content.py --content CONTENT_ID [--dry-run] [--verbose]
"""

import sys
import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime as dt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(get_project_root()))

from src.processing.pipeline_manager import PipelineManager
from src.database.session import get_session
from src.database.models import (
    Content, TaskQueue, ContentChunk, SpeakerTranscription, EmbeddingSegment,
    AlternativeTranscription, Sentence
)

# Import ThemeClassification - may be in a different location
try:
    from src.database.models import ThemeClassification
except ImportError:
    # Try alternate import path
    try:
        from src.backend.models import ThemeClassification
    except ImportError:
        ThemeClassification = None
from src.utils.human_behavior import HumanBehaviorManager
from src.utils.logger import setup_worker_logger
from src.storage.s3_utils import S3StorageConfig, S3Storage
from collections import defaultdict
import yaml
import json
import re
import unicodedata

logger = setup_worker_logger('audit_content')


class TranscriptQualityEvaluator:
    """Evaluates transcript quality and identifies issues."""

    def __init__(self, s3_storage: S3Storage):
        self.s3_storage = s3_storage

    def evaluate_content_transcripts(self, content_id: str, session) -> Dict[str, Any]:
        """
        Evaluate transcript quality for a content item.

        Checks for:
        1. Missing accents/special characters
        2. Word spacing issues
        3. Timing misalignment between speaker turns and embedding segments
        4. Truncated or corrupted transcripts
        5. Oversized embedding segments (>400 tokens, >3min duration)

        Returns:
            Dict with quality metrics and issues found
        """
        results = {
            'content_id': content_id,
            'issues': [],
            'metrics': {},
            'severity': 'none',  # none, low, medium, high, critical
            'recommendations': []
        }

        try:
            # Get content record
            content = session.query(Content).filter_by(content_id=content_id).first()
            if not content:
                results['issues'].append({'type': 'missing_content', 'message': 'Content not found in database'})
                results['severity'] = 'critical'
                return results

            # Check if content has transcripts
            # Also check for embedding_segments - transcription may have happened even if flag is wrong
            has_segments = session.query(EmbeddingSegment).filter_by(content_id=content.id).first() is not None
            if not content.is_transcribed and not has_segments:
                results['issues'].append({'type': 'not_transcribed', 'message': 'Content is not transcribed'})
                results['severity'] = 'high'
                return results

            # Run all quality checks with individual error handling
            # 1. Check source transcript chunks for accent/character issues
            try:
                accent_issues = self._check_accent_preservation(content_id, content)
                if accent_issues:
                    results['issues'].extend(accent_issues)
            except Exception as e:
                logger.warning(f"Failed to check accent preservation: {e}")

            # 2. Check speaker transcriptions for word spacing issues
            try:
                spacing_issues = self._check_word_spacing(session, content)
                if spacing_issues:
                    results['issues'].extend(spacing_issues)
            except Exception as e:
                logger.warning(f"Failed to check word spacing: {e}")

            # 3. Check timing alignment between speaker turns and embedding segments
            try:
                timing_issues = self._check_timing_alignment(session, content)
                if timing_issues:
                    results['issues'].extend(timing_issues)
            except Exception as e:
                logger.warning(f"Failed to check timing alignment: {e}")

            # 4. Check for truncated/corrupted transcripts
            try:
                corruption_issues = self._check_transcript_corruption(session, content)
                if corruption_issues:
                    results['issues'].extend(corruption_issues)
            except Exception as e:
                logger.warning(f"Failed to check transcript corruption: {e}")

            # 5. Check for segments that are too long
            try:
                segment_length_issues = self._check_segment_lengths(session, content)
                if segment_length_issues:
                    results['issues'].extend(segment_length_issues)
            except Exception as e:
                logger.warning(f"Failed to check segment lengths: {e}")

            # 6. Check for language mismatch (e.g., French audio transcribed as English)
            try:
                language_issues = self._check_language_mismatch(session, content)
                if language_issues:
                    results['issues'].extend(language_issues)
            except Exception as e:
                logger.warning(f"Failed to check language mismatch: {e}")

            # Calculate metrics
            results['metrics'] = self._calculate_quality_metrics(results['issues'])

            # Determine overall severity
            results['severity'] = self._determine_severity(results['issues'])

            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results['issues'])

        except Exception as e:
            logger.error(f"Error evaluating transcript quality for {content_id}: {e}", exc_info=True)
            results['issues'].append({
                'type': 'evaluation_error',
                'message': f'Failed to evaluate: {str(e)}'
            })
            results['severity'] = 'unknown'

        return results

    def _check_accent_preservation(self, content_id: str, content) -> List[Dict[str, Any]]:
        """Check if accents and special characters are preserved in source transcripts."""
        issues = []

        try:
            # Sample a few transcript chunks from S3
            chunk_indices = [0, 1, 2] if content.duration > 300 else [0]

            for chunk_idx in chunk_indices:
                transcript_key = f"content/{content_id}/chunks/{chunk_idx}/transcript_words.json.gz"

                if not self.s3_storage.file_exists(transcript_key):
                    transcript_key = transcript_key.replace('.gz', '')

                if self.s3_storage.file_exists(transcript_key):
                    transcript_data = self.s3_storage.read_json_flexible(transcript_key)

                    if transcript_data and 'words' in transcript_data:
                        # Check for ASCII-only text in languages that should have accents
                        text_sample = ' '.join([w.get('word', '') for w in transcript_data['words'][:100]])

                        # Detect if text is ASCII-only (potential accent stripping)
                        is_ascii_only = all(ord(char) < 128 for char in text_sample)

                        # Check metadata for language
                        language = transcript_data.get('language', 'unknown')

                        # Languages that commonly use accents
                        accent_languages = ['fr', 'es', 'pt', 'de', 'it', 'pl', 'cs', 'sv', 'no']

                        if language in accent_languages and is_ascii_only and len(text_sample) > 50:
                            issues.append({
                                'type': 'missing_accents',
                                'severity': 'medium',
                                'chunk_index': chunk_idx,
                                'language': language,
                                'message': f'Chunk {chunk_idx}: Text appears to be ASCII-only for language "{language}" which typically uses accents',
                                'sample': text_sample[:200]
                            })

                        # Check for mojibake patterns (encoding issues)
                        mojibake_patterns = ['Ã©', 'Ã¨', 'Ã ', 'â€™', 'â€œ']
                        if any(pattern in text_sample for pattern in mojibake_patterns):
                            issues.append({
                                'type': 'encoding_corruption',
                                'severity': 'high',
                                'chunk_index': chunk_idx,
                                'message': f'Chunk {chunk_idx}: Detected mojibake/encoding corruption patterns',
                                'sample': text_sample[:200]
                            })

        except Exception as e:
            logger.warning(f"Error checking accent preservation for {content_id}: {e}")

        return issues

    def _check_word_spacing(self, session, content) -> List[Dict[str, Any]]:
        """Check for word spacing issues in sentences."""
        issues = []

        try:
            # Sample sentences (use integer primary key, not string content_id)
            sentences = session.query(Sentence).filter_by(
                content_id=content.id
            ).order_by(Sentence.sentence_index).limit(100).all()

            spacing_problems = 0
            samples = []

            for sent in sentences:
                if not sent.text:
                    continue

                # Check for consecutive words without spaces
                # Pattern: letter followed by capital letter without space
                no_space_pattern = r'[a-zà-ÿ][A-ZÀ-Ÿ]'
                matches = re.findall(no_space_pattern, sent.text)

                # Check for excessive concatenation
                # Words longer than 20 chars without spaces might be concatenated
                long_words = [w for w in sent.text.split() if len(w) > 25]

                if matches or long_words:
                    spacing_problems += 1
                    if len(samples) < 3:  # Keep first 3 samples
                        samples.append({
                            'sentence_index': sent.sentence_index,
                            'turn_index': sent.turn_index,
                            'text': sent.text[:200],
                            'matches': len(matches),
                            'long_words': len(long_words)
                        })

            if spacing_problems > 0:
                severity = 'critical' if spacing_problems > len(sentences) * 0.5 else 'high' if spacing_problems > 3 else 'medium'
                issues.append({
                    'type': 'word_spacing',
                    'severity': severity,
                    'affected_sentences': spacing_problems,
                    'total_sentences_checked': len(sentences),
                    'message': f'Found {spacing_problems} sentences with potential word spacing issues',
                    'samples': samples
                })

        except Exception as e:
            logger.warning(f"Error checking word spacing for {content.content_id}: {e}")

        return issues

    def _check_timing_alignment(self, session, content) -> List[Dict[str, Any]]:
        """Check timing alignment and content coverage between sentences and embedding segments.

        This check verifies:
        1. No large gaps between consecutive segments
        2. All sentences are referenced by at least one segment
        3. Segment coverage spans the full content duration
        """
        issues = []

        try:
            # Get sentences (use integer primary key, not string content_id)
            sentences = session.query(Sentence).filter_by(
                content_id=content.id
            ).order_by(Sentence.sentence_index).all()

            # Get embedding segments
            segments = session.query(EmbeddingSegment).filter_by(
                content_id=content.id
            ).order_by(EmbeddingSegment.start_time).all()

            if not segments or not sentences:
                return issues

            # Build a lookup dict for sentences by index
            sentence_by_idx = {sent.sentence_index: sent for sent in sentences}

            # Check 1: Verify no large gaps between consecutive segments
            gap_issues = []
            for i in range(len(segments) - 1):
                current_seg = segments[i]
                next_seg = segments[i + 1]

                gap = next_seg.start_time - current_seg.end_time

                # Flag gaps larger than 10 seconds (likely missing content)
                if gap > 10.0:
                    gap_issues.append({
                        'segment_indices': f'{current_seg.segment_index}-{next_seg.segment_index}',
                        'gap_duration': f'{gap:.1f}s',
                        'gap_location': f'{current_seg.end_time:.1f}s-{next_seg.start_time:.1f}s'
                    })

            if gap_issues:
                severity = 'high' if len(gap_issues) > 5 else 'medium'
                issues.append({
                    'type': 'segment_gaps',
                    'severity': severity,
                    'gap_count': len(gap_issues),
                    'total_segments': len(segments),
                    'message': f'Found {len(gap_issues)} gaps >10s between consecutive segments',
                    'samples': gap_issues[:3]
                })

            # Check 2: Verify all sentences are referenced by at least one segment
            all_referenced_sentence_ids = set()
            for segment in segments:
                if segment.source_sentence_ids:
                    all_referenced_sentence_ids.update(segment.source_sentence_ids)

            all_sentence_indices = {sent.sentence_index for sent in sentences}
            unreferenced_indices = all_sentence_indices - all_referenced_sentence_ids

            if unreferenced_indices:
                # Get details of unreferenced sentences
                unreferenced_sentences = [sentence_by_idx[idx] for idx in unreferenced_indices if idx in sentence_by_idx]

                # Calculate total duration of unreferenced content
                total_unreferenced_duration = sum(
                    sent.end_time - sent.start_time for sent in unreferenced_sentences
                )

                severity = 'critical' if total_unreferenced_duration > 60 else 'high' if total_unreferenced_duration > 10 else 'medium'

                samples = []
                for sent in sorted(unreferenced_sentences, key=lambda s: s.sentence_index)[:3]:
                    samples.append({
                        'sentence_index': sent.sentence_index,
                        'turn_index': sent.turn_index,
                        'time_range': f'{sent.start_time:.1f}s-{sent.end_time:.1f}s',
                        'duration': f'{sent.end_time - sent.start_time:.1f}s',
                        'text_preview': sent.text[:100] if sent.text else 'Empty'
                    })

                issues.append({
                    'type': 'unreferenced_sentences',
                    'severity': severity,
                    'unreferenced_count': len(unreferenced_indices),
                    'total_sentences': len(sentences),
                    'unreferenced_duration': f'{total_unreferenced_duration:.1f}s',
                    'message': f'Found {len(unreferenced_indices)} sentences not referenced by any segment ({total_unreferenced_duration:.1f}s of content)',
                    'samples': samples
                })

            # Check 3: Verify segment coverage spans the full content duration
            if segments and sentences:
                first_sentence_start = min(s.start_time for s in sentences)
                last_sentence_end = max(s.end_time for s in sentences)

                first_segment_start = segments[0].start_time
                last_segment_end = segments[-1].end_time

                # Check if segments start significantly after first sentence (>5s)
                start_gap = first_segment_start - first_sentence_start
                if start_gap > 5.0:
                    issues.append({
                        'type': 'missing_start_content',
                        'severity': 'high',
                        'gap_duration': f'{start_gap:.1f}s',
                        'first_sentence_start': f'{first_sentence_start:.1f}s',
                        'first_segment_start': f'{first_segment_start:.1f}s',
                        'message': f'Segments start {start_gap:.1f}s after first sentence - missing content at beginning'
                    })

                # Check if segments end significantly before last sentence (>5s)
                end_gap = last_sentence_end - last_segment_end
                if end_gap > 5.0:
                    issues.append({
                        'type': 'missing_end_content',
                        'severity': 'high',
                        'gap_duration': f'{end_gap:.1f}s',
                        'last_sentence_end': f'{last_sentence_end:.1f}s',
                        'last_segment_end': f'{last_segment_end:.1f}s',
                        'message': f'Segments end {end_gap:.1f}s before last sentence - missing content at end'
                    })

        except Exception as e:
            logger.warning(f"Error checking timing alignment for {content.content_id}: {e}")

        return issues

    def _check_transcript_corruption(self, session, content) -> List[Dict[str, Any]]:
        """Check for truncated or corrupted transcripts."""
        issues = []

        try:
            # Check if number of speaker turns is reasonable given duration
            # Count unique turns from sentences table (turn_index)
            from sqlalchemy import func
            turn_count_result = session.query(func.count(func.distinct(Sentence.turn_index))).filter_by(
                content_id=content.id
            ).scalar()
            speaker_turns = turn_count_result or 0

            # Also get sentence count for additional context
            sentence_count = session.query(Sentence).filter_by(content_id=content.id).count()

            # Expect roughly 1 turn per 80 seconds minimum (very relaxed threshold)
            # This catches truly broken transcriptions (0 or very few turns) without flagging normal content
            expected_min_turns = max(1, int(content.duration / 80))
            expected_max_turns = int(content.duration / 5)

            if speaker_turns < expected_min_turns:
                issues.append({
                    'type': 'insufficient_turns',
                    'severity': 'high',
                    'speaker_turns': speaker_turns,
                    'sentence_count': sentence_count,
                    'expected_min': expected_min_turns,
                    'duration': content.duration,
                    'message': f'Only {speaker_turns} speaker turns for {content.duration:.0f}s duration (expected ≥{expected_min_turns})'
                })

            # Check if chunks exist but have no transcripts
            # Skip this check if content is stitched - stitched content has transcripts in database
            if not content.is_stitched or speaker_turns == 0:
                chunks = session.query(ContentChunk).filter_by(
                    content_id=content.id,
                    transcription_status='completed'
                ).all()

                empty_chunks = []
                for chunk in chunks[:10]:  # Check first 10 chunks
                    transcript_key = f"content/{content.content_id}/chunks/{chunk.chunk_index}/transcript_words.json.gz"
                    if not self.s3_storage.file_exists(transcript_key):
                        transcript_key = transcript_key.replace('.gz', '')

                    if self.s3_storage.file_exists(transcript_key):
                        transcript_data = self.s3_storage.read_json_flexible(transcript_key)
                        # Check for both new format (words) and old format (segments)
                        has_words = transcript_data and transcript_data.get('words')
                        has_segments = transcript_data and transcript_data.get('segments')
                        if not (has_words or has_segments):
                            empty_chunks.append(chunk.chunk_index)
                    else:
                        # File doesn't exist at all - only flag if content isn't stitched
                        if not content.is_stitched:
                            empty_chunks.append(chunk.chunk_index)

                if empty_chunks:
                    issues.append({
                        'type': 'empty_transcripts',
                        'severity': 'high',
                        'empty_chunks': empty_chunks,
                        'message': f'Found {len(empty_chunks)} chunks marked as transcribed but with empty transcripts'
                    })

        except Exception as e:
            logger.warning(f"Error checking transcript corruption for {content.content_id}: {e}")

        return issues

    def _check_segment_lengths(self, session, content) -> List[Dict[str, Any]]:
        """Check for embedding segments that are too long."""
        issues = []

        try:
            # Get embedding segments
            segments = session.query(EmbeddingSegment).filter_by(
                content_id=content.id
            ).all()

            if not segments:
                return issues

            # Define thresholds for segment issues
            # Token count thresholds (typical target is 250 tokens)
            MAX_TOKENS_NORMAL = 500  # Warn if over this
            MAX_TOKENS_CRITICAL = 700  # Critical if over this

            # Duration thresholds (typical segments should be 30-120 seconds)
            MAX_DURATION_NORMAL = 180  # 3 minutes - warn
            MAX_DURATION_CRITICAL = 300  # 5 minutes - critical

            # Text length thresholds (rough char count)
            MAX_CHARS_NORMAL = 2000
            MAX_CHARS_CRITICAL = 3500

            oversized_segments = []
            critically_oversized = []
            samples = []

            for segment in segments:
                is_oversized = False
                is_critical = False
                reasons = []

                # Check token count
                if segment.token_count > MAX_TOKENS_CRITICAL:
                    is_oversized = True
                    is_critical = True
                    reasons.append(f'{segment.token_count} tokens (critical, >{MAX_TOKENS_CRITICAL})')
                elif segment.token_count > MAX_TOKENS_NORMAL:
                    is_oversized = True
                    reasons.append(f'{segment.token_count} tokens (high, >{MAX_TOKENS_NORMAL})')

                # Check duration
                duration = segment.end_time - segment.start_time
                if duration > MAX_DURATION_CRITICAL:
                    is_oversized = True
                    is_critical = True
                    reasons.append(f'{duration:.1f}s duration (critical, >5min)')
                elif duration > MAX_DURATION_NORMAL:
                    is_oversized = True
                    reasons.append(f'{duration:.1f}s duration (high, >3min)')

                # Check text length
                text_len = len(segment.text) if segment.text else 0
                if text_len > MAX_CHARS_CRITICAL:
                    is_oversized = True
                    is_critical = True
                    reasons.append(f'{text_len} chars (critical, >3500)')
                elif text_len > MAX_CHARS_NORMAL:
                    is_oversized = True
                    reasons.append(f'{text_len} chars (high, >2000)')

                if is_oversized:
                    oversized_segments.append(segment.segment_index)
                    if is_critical:
                        critically_oversized.append(segment.segment_index)

                    if len(samples) < 5:  # Keep first 5 samples
                        samples.append({
                            'segment_index': segment.segment_index,
                            'token_count': segment.token_count,
                            'duration': f'{duration:.1f}s',
                            'char_count': text_len,
                            'segment_type': segment.segment_type,
                            'reasons': ', '.join(reasons),
                            'text_preview': segment.text[:150] + '...' if text_len > 150 else segment.text
                        })

            if oversized_segments:
                # Calculate severity based on prevalence
                total_segments = len(segments)
                oversized_pct = len(oversized_segments) / total_segments
                critical_pct = len(critically_oversized) / total_segments

                if critical_pct > 0.1:  # More than 10% critically oversized
                    severity = 'critical'
                elif critical_pct > 0 or oversized_pct > 0.3:  # Any critical or >30% oversized
                    severity = 'high'
                elif oversized_pct > 0.1:  # More than 10% oversized
                    severity = 'medium'
                else:
                    severity = 'low'

                issues.append({
                    'type': 'oversized_segments',
                    'severity': severity,
                    'oversized_count': len(oversized_segments),
                    'critically_oversized_count': len(critically_oversized),
                    'total_segments': total_segments,
                    'oversized_percentage': f'{oversized_pct * 100:.1f}%',
                    'message': f'Found {len(oversized_segments)} oversized segments ({oversized_pct * 100:.1f}% of {total_segments} total)',
                    'samples': samples
                })

        except Exception as e:
            logger.warning(f"Error checking segment lengths for {content.content_id}: {e}")

        return issues

    def _check_language_mismatch(self, session, content) -> List[Dict[str, Any]]:
        """Check if transcript language matches expected content language.

        Detects cases where audio in one language was transcribed as another
        (e.g., French audio transcribed with English Whisper producing gibberish).
        """
        issues = []

        try:
            # Only check non-English content (English is the default/fallback)
            expected_lang = content.main_language
            if not expected_lang or expected_lang == 'en':
                return issues

            # Get sample embedding segments to analyze (include segment 0 for short content)
            segments = session.query(EmbeddingSegment).filter_by(
                content_id=content.id
            ).filter(
                EmbeddingSegment.segment_index.between(0, 10)
            ).all()

            if not segments:
                return issues

            # Language detection word lists
            french_words = [' le ', ' la ', ' les ', ' des ', ' que ', ' qui ',
                          ' est ', ' une ', ' pas ', ' pour ', ' dans ', ' avec ',
                          ' sur ', ' sont ', ' fait ', ' être ', ' avoir ']
            english_words = [' the ', ' and ', ' but ', ' have ', ' with ', ' that ',
                           ' this ', ' they ', ' was ', ' are ', ' been ', ' would ',
                           ' could ', ' should ', ' from ', ' which ', ' their ']
            spanish_words = [' el ', ' la ', ' los ', ' las ', ' que ', ' de ',
                           ' en ', ' un ', ' una ', ' por ', ' para ', ' con ']

            # Count word occurrences across segments
            total_french = 0
            total_english = 0
            total_spanish = 0
            total_text_length = 0
            samples = []

            for segment in segments:
                if not segment.text or len(segment.text) < 50:
                    continue

                text_lower = segment.text.lower()
                total_text_length += len(text_lower)

                # Count language indicators
                for word in french_words:
                    total_french += text_lower.count(word)
                for word in english_words:
                    total_english += text_lower.count(word)
                for word in spanish_words:
                    total_spanish += text_lower.count(word)

                # Keep samples for reporting
                if len(samples) < 3:
                    samples.append({
                        'segment_index': segment.segment_index,
                        'text_preview': segment.text[:150] + '...' if len(segment.text) > 150 else segment.text
                    })

            if total_text_length < 500:
                # Not enough text to analyze
                return issues

            # Determine detected language
            scores = {
                'fr': total_french,
                'en': total_english,
                'es': total_spanish
            }
            detected_lang = max(scores, key=scores.get)
            detected_score = scores[detected_lang]
            expected_score = scores.get(expected_lang, 0)

            # Check for mismatch
            # Criteria: English score is high AND/OR expected language score is low
            if expected_lang == 'fr':
                # French content should have high French score, low English score
                if total_english > 15 and total_french < 10:
                    # Clear garbage: lots of English, almost no French
                    severity = 'critical'
                    mismatch_type = 'definite'
                elif total_english > total_french * 0.5 and total_english > 8:
                    # English dominates or is close to French
                    severity = 'high'
                    mismatch_type = 'likely'
                elif total_english > 80 and total_french < total_english * 2:
                    # Very high English presence relative to French
                    # This catches mixed garbage transcripts while allowing for English ads/clips
                    # (80+ English words AND French is less than 2x English)
                    severity = 'medium'
                    mismatch_type = 'suspected'
                else:
                    return issues

                issues.append({
                    'type': 'language_mismatch',
                    'severity': severity,
                    'expected_language': expected_lang,
                    'detected_language': 'en',
                    'mismatch_type': mismatch_type,
                    'french_score': total_french,
                    'english_score': total_english,
                    'message': f'{mismatch_type.title()} language mismatch: expected {expected_lang}, detected English (fr_score={total_french}, en_score={total_english})',
                    'samples': samples
                })

            elif expected_lang == 'es':
                # Spanish content check
                if total_english > 15 and total_spanish < 10:
                    issues.append({
                        'type': 'language_mismatch',
                        'severity': 'critical',
                        'expected_language': expected_lang,
                        'detected_language': 'en',
                        'mismatch_type': 'definite',
                        'spanish_score': total_spanish,
                        'english_score': total_english,
                        'message': f'Language mismatch: expected {expected_lang}, detected English',
                        'samples': samples
                    })

        except Exception as e:
            logger.warning(f"Error checking language mismatch for {content.content_id}: {e}")

        return issues

    def _calculate_quality_metrics(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall quality metrics from issues."""
        metrics = {
            'total_issues': len(issues),
            'by_type': defaultdict(int),
            'by_severity': defaultdict(int)
        }

        for issue in issues:
            issue_type = issue.get('type', 'unknown')
            severity = issue.get('severity', 'unknown')
            metrics['by_type'][issue_type] += 1
            metrics['by_severity'][severity] += 1

        return dict(metrics)

    def _determine_severity(self, issues: List[Dict[str, Any]]) -> str:
        """Determine overall severity from issues list."""
        if not issues:
            return 'none'

        severities = [issue.get('severity', 'unknown') for issue in issues]

        if 'critical' in severities:
            return 'critical'
        elif 'high' in severities:
            return 'high'
        elif 'medium' in severities:
            return 'medium'
        elif 'low' in severities:
            return 'low'
        else:
            return 'unknown'

    def _generate_recommendations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on issues found."""
        recommendations = []
        issue_types = {issue.get('type') for issue in issues}

        if 'missing_accents' in issue_types or 'encoding_corruption' in issue_types:
            recommendations.append('Re-transcribe content with proper UTF-8 encoding to preserve accents and special characters')

        if 'word_spacing' in issue_types:
            recommendations.append('Re-stitch content to fix word spacing issues in speaker turns')

        if 'segment_gaps' in issue_types:
            recommendations.append('Re-segment embeddings - large gaps detected between consecutive segments')

        if 'unreferenced_turns' in issue_types:
            # Get duration of unreferenced content
            unreferenced_issue = next((i for i in issues if i.get('type') == 'unreferenced_turns'), None)
            if unreferenced_issue:
                duration = unreferenced_issue.get('unreferenced_duration', '0s')
                recommendations.append(f'Re-segment embeddings - {unreferenced_issue.get("unreferenced_count", 0)} speaker turns ({duration}) not referenced by any segment')
            else:
                recommendations.append('Re-segment embeddings - some speaker turns not referenced by any segment')

        if 'missing_start_content' in issue_types or 'missing_end_content' in issue_types:
            recommendations.append('Re-segment embeddings - segments do not span full content duration')

        if 'insufficient_turns' in issue_types or 'empty_transcripts' in issue_types:
            recommendations.append('Re-transcribe content - possible transcription failure or corruption')

        if 'oversized_segments' in issue_types:
            # Check severity to provide specific guidance
            oversized_issue = next((i for i in issues if i.get('type') == 'oversized_segments'), None)
            if oversized_issue:
                critically_oversized = oversized_issue.get('critically_oversized_count', 0)
                if critically_oversized > 0:
                    recommendations.append(f'Re-segment embeddings to split {critically_oversized} critically oversized segments (>600 tokens or >5min)')
                else:
                    recommendations.append('Re-segment embeddings with stricter length constraints to reduce segment sizes')

        if 'language_mismatch' in issue_types:
            # Get details from the issue
            lang_issue = next((i for i in issues if i.get('type') == 'language_mismatch'), None)
            if lang_issue:
                expected = lang_issue.get('expected_language', 'unknown')
                detected = lang_issue.get('detected_language', 'unknown')
                mismatch_type = lang_issue.get('mismatch_type', 'detected')
                recommendations.append(
                    f'Re-transcribe content with correct language ({expected}) - '
                    f'{mismatch_type} mismatch detected (transcribed as {detected})'
                )

        return recommendations


class ContentAuditor:
    """Audits content state and creates missing tasks."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the auditor with configuration."""
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Initialize dependencies
        self.worker_task_failures = defaultdict(dict)
        self.behavior_manager = HumanBehaviorManager(config_path)
        self.pipeline_manager = PipelineManager(
            self.behavior_manager, 
            self.config, 
            self.worker_task_failures
        )
        
        # Initialize S3 storage for diarization method detection
        s3_config = S3StorageConfig(
            endpoint_url=self.config['storage']['s3']['endpoint_url'],
            access_key=self.config['storage']['s3']['access_key'],
            secret_key=self.config['storage']['s3']['secret_key'],
            bucket_name=self.config['storage']['s3']['bucket_name'],
            use_ssl=self.config['storage']['s3']['use_ssl']
        )
        self.s3_storage = S3Storage(s3_config)

        # Initialize transcript quality evaluator
        self.quality_evaluator = TranscriptQualityEvaluator(self.s3_storage)

        logger.info("ContentAuditor initialized")
    
    def detect_diarization_method_from_file(self, diarization_s3_path: str) -> Optional[str]:
        """Detect diarization method by analyzing the diarization.json file structure."""
        try:
            # Use read_json_flexible which handles both compressed and uncompressed
            diarization_data = self.s3_storage.read_json_flexible(diarization_s3_path)
            if not diarization_data:
                logger.warning(f"Could not read diarization file from {diarization_s3_path}")
                return None
            
            # Analyze structure to determine method
            if isinstance(diarization_data, dict):
                # Check if it has an explicit method field first (most reliable)
                if 'method' in diarization_data:
                    return diarization_data['method']
                elif 'diarization_method' in diarization_data:
                    return diarization_data['diarization_method']
                
                # Check for FluidAudio characteristics
                elif 'speakerEmbeddings' in diarization_data or 'speaker_centroids' in diarization_data:
                    # FluidAudio includes speaker embeddings/centroids
                    return 'fluid_audio'
                
                # Check for PyAnnote characteristics  
                elif 'speakers' in diarization_data or ('segments' in diarization_data and 'method' not in diarization_data):
                    # Standard PyAnnote structure (only if method field is not present)
                    return 'pyannote3.1'
                
            # If we can't determine from structure, return None
            logger.warning(f"Could not determine diarization method from file structure in {diarization_s3_path}")
            return None
            
        except Exception as e:
            logger.warning(f"Error detecting diarization method from {diarization_s3_path}: {str(e)}")
            return None

    def get_content_summary(self, session, content: Content) -> Dict[str, Any]:
        """Get a summary of content's current state."""
        # Get chunk information
        chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
        chunk_summary = {
            'total_chunks': len(chunks),
            'extracted_chunks': len([c for c in chunks if c.extraction_status == 'completed']),
            'transcribed_chunks': len([c for c in chunks if c.transcription_status == 'completed']),
            'chunk_details': [
                {
                    'index': c.chunk_index,
                    'extraction_status': c.extraction_status,
                    'transcription_status': c.transcription_status,
                    'duration': c.duration
                }
                for c in sorted(chunks, key=lambda x: x.chunk_index)
            ]
        }
        
        # Get pending tasks
        pending_tasks = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.status.in_(['pending', 'processing'])
        ).all()
        
        # Get recent failed tasks
        failed_tasks = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.status == 'failed'
        ).order_by(TaskQueue.created_at.desc()).limit(5).all()
        
        return {
            'content_id': content.content_id,
            'platform': content.platform,
            'title': content.title[:100] + '...' if content.title and len(content.title) > 100 else content.title,
            'publish_date': content.publish_date.isoformat() if content.publish_date else None,
            'duration': content.duration,
            'projects': content.projects,
            'flags': {
                'is_downloaded': content.is_downloaded,
                'is_converted': content.is_converted,
                'is_transcribed': content.is_transcribed,
                'is_diarized': content.is_diarized,
                'is_stitched': content.is_stitched,
                'is_embedded': content.is_embedded,
                'is_compressed': content.is_compressed,
                'blocked_download': content.blocked_download
            },
            'versions': {
                'stitch_version': content.stitch_version,
                'segment_version': content.meta_data.get('segment_version') if content.meta_data else None,
                'diarization_method': content.diarization_method
            },
            'chunks': chunk_summary,
            'pending_tasks': [
                {
                    'id': task.id,
                    'task_type': task.task_type,
                    'status': task.status,
                    'priority': task.priority,
                    'created_at': task.created_at.isoformat(),
                    'input_data': task.input_data
                }
                for task in pending_tasks
            ],
            'recent_failed_tasks': [
                {
                    'id': task.id,
                    'task_type': task.task_type,
                    'error': task.error[:200] + '...' if task.error and len(task.error) > 200 else task.error,
                    'completed_at': task.completed_at.isoformat() if task.completed_at else None
                }
                for task in failed_tasks
            ],
            'meta_data': content.meta_data,
            'last_updated': content.last_updated.isoformat() if content.last_updated else None
        }
    
    def check_corrupted_source_audio(self, content_id: str) -> bool:
        """Check if source file has corrupted/empty audio using ffprobe with presigned URL.

        Returns True if corruption detected, False otherwise.
        """
        try:
            import subprocess

            # Check for source file in S3
            source_formats = ['source.mp4', 'source.webm', 'source.mkv', 'source.m4a', 'source.mp3']
            source_key = None

            for fmt in source_formats:
                test_key = f"content/{content_id}/{fmt}"
                if self.s3_storage.file_exists(test_key):
                    source_key = test_key
                    break

            if not source_key:
                logger.debug(f"[{content_id}] No source file found in S3")
                return False

            # Get presigned URL to probe without downloading
            presigned_url = self.s3_storage.get_file_url(source_key, expires_in=300)  # 5 min expiry
            if not presigned_url:
                logger.warning(f"[{content_id}] Could not get presigned URL for source file")
                return False

            # Use ffprobe to check audio streams without downloading
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=codec_type,duration,nb_frames',
                 '-of', 'json', presigned_url],
                capture_output=True,
                text=True,
                timeout=15
            )

            if result.returncode != 0:
                logger.warning(f"[{content_id}] ffprobe failed on source: {result.stderr[-200:]}")
                return True

            # Parse ffprobe output
            try:
                probe_data = json.loads(result.stdout)
                streams = probe_data.get('streams', [])

                if not streams:
                    logger.warning(f"[{content_id}] No audio streams found in source file")
                    return True

                # Check if audio stream has duration
                audio_stream = streams[0]
                duration = float(audio_stream.get('duration', 0))

                if duration <= 0.1:  # Less than 0.1 seconds = corrupted
                    logger.warning(f"[{content_id}] Source audio has invalid duration ({duration}s)")
                    return True

                logger.debug(f"[{content_id}] Source audio verified OK (duration: {duration}s)")
                return False

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"[{content_id}] Could not parse ffprobe output: {e}")
                return True

        except subprocess.TimeoutExpired:
            logger.warning(f"[{content_id}] ffprobe timeout - source may be corrupted")
            return True
        except Exception as e:
            logger.error(f"[{content_id}] Error checking source audio: {str(e)}")
            return False

    async def audit_content(self, content_id: str, dry_run: bool = False, verbose: bool = False,
                           skip_quality_check: bool = False, fix_issues: bool = False,
                           skip_state_reconciliation: bool = False) -> Dict[str, Any]:
        """Audit a specific content item.

        Args:
            content_id: Content ID to audit
            dry_run: If True, only show what would be done
            verbose: If True, show detailed output
            skip_quality_check: If True, skip transcript quality evaluation
            fix_issues: If True, fix detected critical issues (e.g., language mismatch)
            skip_state_reconciliation: If True, skip state reconciliation (no task creation for stitch/embed etc)
        """
        logger.info(f"Starting audit for content {content_id} (dry_run={dry_run}, fix_issues={fix_issues})")

        results = {
            'content_id': content_id,
            'audit_timestamp': dt.now().isoformat(),
            'dry_run': dry_run,
            'fix_issues': fix_issues,
            'status': 'success',
            'errors': [],
            'before_state': None,
            'after_state': None,
            'evaluation_results': None,
            'transcript_quality': None,
            'actions_taken': [],
            'tasks_deleted': [],
            'corrupted_files_deleted': [],
            'issues_fixed': []
        }

        try:
            with get_session() as session:
                # Get content
                logger.info(f"[{content_id}] Querying content from database...")
                content = session.query(Content).filter_by(content_id=content_id).first()
                logger.info(f"[{content_id}] Content found: {content is not None}")
                if not content:
                    results['status'] = 'error'
                    results['errors'].append(f"Content {content_id} not found in database")
                    return results

                # Capture before state
                logger.info(f"[{content_id}] Getting content summary...")
                results['before_state'] = self.get_content_summary(session, content)
                logger.info(f"[{content_id}] Content summary retrieved")

                if verbose:
                    logger.info(f"Before state: {json.dumps(results['before_state']['flags'], indent=2)}")

                # Evaluate transcript quality if content is transcribed or has embedding segments (unless skipped)
                # Check for embedding_segments even if is_transcribed=False (may be incorrectly set)
                has_segments = session.query(EmbeddingSegment).filter_by(content_id=content.id).first() is not None
                if (content.is_transcribed or has_segments) and not skip_quality_check:
                    logger.info(f"[{content_id}] Evaluating transcript quality...")
                    try:
                        quality_results = self.quality_evaluator.evaluate_content_transcripts(content_id, session)
                        results['transcript_quality'] = quality_results

                        if quality_results['severity'] not in ['none', 'low']:
                            logger.warning(f"[{content_id}] Transcript quality issues detected: {quality_results['severity']} severity")
                            if verbose:
                                logger.info(f"Quality issues: {json.dumps(quality_results['issues'], indent=2)}")
                    except Exception as e:
                        logger.error(f"[{content_id}] Quality evaluation failed: {e}", exc_info=True)
                        results['transcript_quality'] = {
                            'content_id': content_id,
                            'issues': [],
                            'metrics': {},
                            'severity': 'unknown',
                            'recommendations': [],
                            'error': str(e)
                        }

                # Fix language mismatch if detected and fix_issues is enabled
                if fix_issues and results.get('transcript_quality'):
                    quality = results['transcript_quality']
                    language_mismatch_issues = [
                        i for i in quality.get('issues', [])
                        if i.get('type') == 'language_mismatch'
                    ]

                    if language_mismatch_issues:
                        issue = language_mismatch_issues[0]
                        logger.warning(f"[{content_id}] Language mismatch detected - {'would fix' if dry_run else 'fixing'}...")

                        fix_result = self.fix_language_mismatch(session, content, dry_run=dry_run)
                        results['issues_fixed'].append({
                            'type': 'language_mismatch',
                            'fix_result': fix_result
                        })

                        if fix_result['success']:
                            if dry_run:
                                action = (f"Would fix language mismatch: delete "
                                         f"{fix_result['deleted']['embedding_segments']} segments, "
                                         f"{fix_result['deleted']['sentences']} sentences, "
                                         f"{fix_result['deleted']['theme_classifications']} classifications, "
                                         f"reset flags, create transcribe task")
                            else:
                                action = (f"Fixed language mismatch: deleted "
                                         f"{fix_result['deleted']['embedding_segments']} segments, "
                                         f"{fix_result['deleted']['sentences']} sentences, "
                                         f"{fix_result['deleted']['theme_classifications']} classifications, "
                                         f"created {fix_result['task_created']}")
                            results['actions_taken'].append(action)
                            logger.info(action)
                        else:
                            error_msg = f"Failed to fix language mismatch: {fix_result.get('errors', [])}"
                            results['errors'].append(error_msg)
                            logger.error(error_msg)

                # Check for corrupted source files if downloaded but not fully processed
                if content.is_downloaded and not content.is_compressed:
                    logger.info(f"[{content_id}] Checking for corrupted source audio...")
                    is_corrupted = self.check_corrupted_source_audio(content_id)

                    if is_corrupted:
                        logger.warning(f"[{content_id}] Detected corrupted source audio - will reset download")

                        if not dry_run:
                            # Delete corrupted files from S3
                            files_to_delete = [
                                f"content/{content_id}/source.mp4",
                                f"content/{content_id}/source.webm",
                                f"content/{content_id}/source.mkv",
                                f"content/{content_id}/audio.wav",
                                f"content/{content_id}/audio.opus",
                                f"content/{content_id}/audio.mp3"
                            ]

                            for s3_key in files_to_delete:
                                if self.s3_storage.file_exists(s3_key):
                                    logger.info(f"[{content_id}] Deleting corrupted file: {s3_key}")
                                    if self.s3_storage.delete_file(s3_key):
                                        results['corrupted_files_deleted'].append(s3_key)

                            # Reset download and convert flags
                            content.is_downloaded = False
                            content.is_converted = False
                            content.last_updated = dt.now()
                            session.commit()

                            action = f"Detected corrupted audio - reset download, deleted {len(results['corrupted_files_deleted'])} corrupted files"
                            results['actions_taken'].append(action)
                            logger.info(action)
                        else:
                            action = f"Would reset download and delete corrupted files due to corrupted audio"
                            results['actions_taken'].append(action)

                # Skip state reconciliation if requested (e.g., quality-only audit)
                if skip_state_reconciliation:
                    logger.info(f"[{content_id}] Skipping state reconciliation (quality-only mode)")
                    eval_results = {'flags_updated': [], 'tasks_created': [], 'errors': []}
                elif dry_run:
                    # For dry run, use a separate session to avoid committing changes
                    logger.info("DRY RUN: Evaluating state without making changes...")
                    with get_session() as dry_session:
                        dry_content = dry_session.query(Content).filter_by(content_id=content_id).first()
                        eval_results = await self.pipeline_manager.evaluate_content_state(
                            dry_session, dry_content
                        )
                        # Don't commit the dry run session
                        dry_session.rollback()

                        # Check for inappropriate tasks with the corrected flags (in memory)
                        inappropriate_tasks = self.find_inappropriate_tasks(dry_session, dry_content)
                        if inappropriate_tasks:
                            for task in inappropriate_tasks:
                                results['tasks_deleted'].append(f"{task.task_type} (ID: {task.id})")
                else:
                    # Normal run - actually make changes
                    logger.info("LIVE RUN: Evaluating state and making changes...")
                    eval_results = await self.pipeline_manager.evaluate_content_state(
                        session, content
                    )
                    # Changes are already committed by evaluate_content_state

                    # Now check for inappropriate tasks with the UPDATED flags
                    session.refresh(content)  # Refresh to get updated flags
                    inappropriate_tasks = self.find_inappropriate_tasks(session, content)
                    
                    if inappropriate_tasks:
                        for task in inappropriate_tasks:
                            logger.info(f"Deleting inappropriate {task.task_type} task (ID: {task.id}) after flag update")
                            session.delete(task)
                            results['tasks_deleted'].append(f"{task.task_type} (ID: {task.id})")
                        session.commit()
                
                results['evaluation_results'] = eval_results

                # Check if we need to detect and update diarization method
                # Refresh content to get latest values from all previous commits
                if not dry_run:
                    session.refresh(content)

                if not dry_run and content.is_diarized and not content.diarization_method:
                    diarization_s3_path = f"content/{content.content_id}/diarization.json"
                    
                    # Check if file exists, try compressed version if not
                    if not self.s3_storage.file_exists(diarization_s3_path):
                        diarization_s3_path = f"content/{content.content_id}/diarization.json.gz"
                    
                    if self.s3_storage.file_exists(diarization_s3_path):
                        detected_method = self.detect_diarization_method_from_file(diarization_s3_path)
                        if detected_method:
                            content.diarization_method = detected_method
                            session.commit()
                            action = f"Detected and set diarization method: {detected_method}"
                            results['actions_taken'].append(action)
                            logger.info(action)
                        else:
                            logger.warning(f"Could not detect diarization method for {content_id}")
                    else:
                        logger.warning(f"Diarization file not found for diarized content {content_id}")
                
                # Log what was found
                if results['tasks_deleted']:
                    action = f"{'Would delete' if dry_run else 'Deleted'} inappropriate tasks: {', '.join([t.split(' (')[0] for t in results['tasks_deleted']])}"
                    results['actions_taken'].append(action)
                    logger.info(action)
                
                if eval_results['flags_updated']:
                    action = f"{'Would update' if dry_run else 'Updated'} flags: {', '.join(eval_results['flags_updated'])}"
                    results['actions_taken'].append(action)
                    logger.info(action)
                
                if eval_results['tasks_created']:
                    action = f"{'Would create' if dry_run else 'Created'} tasks: {', '.join(eval_results['tasks_created'])}"
                    results['actions_taken'].append(action)
                    logger.info(action)
                
                if eval_results['errors']:
                    results['errors'].extend(eval_results['errors'])
                    logger.error(f"Evaluation errors: {eval_results['errors']}")
                
                # Capture after state (refresh content first if not dry run)
                if not dry_run:
                    session.refresh(content)
                results['after_state'] = self.get_content_summary(session, content)
                
                if verbose:
                    logger.info(f"After state: {json.dumps(results['after_state']['flags'], indent=2)}")
                
                # Summary
                if not results['actions_taken'] and not results['errors']:
                    results['actions_taken'].append("No changes needed - content state is correct")
                    logger.info("✅ Content state is already correct")
                elif results['actions_taken']:
                    logger.info(f"✅ Completed {len(results['actions_taken'])} actions")
                
                return results
                
        except Exception as e:
            results['status'] = 'error'
            results['errors'].append(f"Audit failed with exception: {str(e)}")
            logger.error(f"Audit failed: {e}", exc_info=True)
            return results

    def find_inappropriate_tasks(self, session, content: Content) -> list:
        """Find pending tasks that are inappropriate given the current content state."""
        inappropriate_tasks = []
        
        # Get all pending tasks for this content
        pending_tasks = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.status.in_(['pending', 'processing'])
        ).all()
        
        logger.info(f"Checking {len(pending_tasks)} pending tasks for content {content.content_id}")
        logger.info(f"Content flags: downloaded={content.is_downloaded}, converted={content.is_converted}, transcribed={content.is_transcribed}, diarized={content.is_diarized}, stitched={content.is_stitched}, embedded={content.is_embedded}, compressed={content.is_compressed}")
        
        for task in pending_tasks:
            task_type = task.task_type
            logger.info(f"Evaluating task {task.id}: type={task_type}, status={task.status}")
            
            # Check if this task is inappropriate based on content flags
            should_delete = False
            reason = ""
            
            if task_type == 'download_youtube' or task_type == 'download_podcast':
                if content.is_downloaded:
                    should_delete = True
                    reason = f"Content already downloaded"
                else:
                    logger.info(f"Task {task.id} ({task_type}) is appropriate - content not downloaded")
            
            elif task_type == 'convert':
                if content.is_converted:
                    should_delete = True
                    reason = f"Content already converted"
                else:
                    logger.info(f"Task {task.id} ({task_type}) is appropriate - content not converted")
            
            elif task_type == 'transcribe':
                if content.is_transcribed:
                    should_delete = True
                    reason = f"Content already transcribed"
                else:
                    logger.info(f"Task {task.id} ({task_type}) is appropriate - content not transcribed")
            
            elif task_type.startswith('diarize'):
                if content.is_diarized:
                    should_delete = True
                    reason = f"Content already diarized"
                else:
                    logger.info(f"Task {task.id} ({task_type}) is appropriate - content not diarized")
            
            elif task_type == 'stitch':
                if content.is_stitched:
                    should_delete = True
                    reason = f"Content already stitched"
                else:
                    logger.info(f"Task {task.id} ({task_type}) is appropriate - content not stitched")
            
            elif task_type == 'embed':
                if content.is_embedded:
                    should_delete = True
                    reason = f"Content already embedded"
                else:
                    logger.info(f"Task {task.id} ({task_type}) is appropriate - content not embedded")
            
            elif task_type == 'cleanup':
                if content.is_compressed:
                    should_delete = True
                    reason = f"Content already compressed"
                else:
                    logger.info(f"Task {task.id} ({task_type}) is appropriate - content not compressed")
            
            else:
                logger.info(f"Task {task.id} ({task_type}) - unknown task type, keeping")
            
            if should_delete:
                logger.info(f"INAPPROPRIATE TASK FOUND: {task_type} task (ID: {task.id}) - {reason}")
                inappropriate_tasks.append(task)
        
        logger.info(f"Found {len(inappropriate_tasks)} inappropriate tasks to delete")
        return inappropriate_tasks

    def fix_language_mismatch(self, session, content: Content, dry_run: bool = False) -> Dict[str, Any]:
        """
        Fix language mismatch by resetting transcription and creating re-transcribe task.

        This deletes all transcription-derived data for THIS CONTENT ONLY:
        - theme_classifications (for this content's segments)
        - alternative_transcriptions (for this content's segments)
        - embedding_segments
        - sentences
        - speaker_transcriptions
        - S3 transcript files

        Then resets content state and creates high-priority transcribe task.

        Args:
            session: Database session
            content: Content object to fix
            dry_run: If True, only report what would be done

        Returns:
            Dict with fix results
        """
        results = {
            'content_id': content.content_id,
            'dry_run': dry_run,
            'success': False,
            'deleted': {
                'theme_classifications': 0,
                'alternative_transcriptions': 0,
                'embedding_segments': 0,
                'sentences': 0,
                'speaker_transcriptions': 0,
                's3_files': 0
            },
            'reset_flags': [],
            'task_created': None,
            'errors': []
        }

        try:
            content_id_int = content.id
            content_id_str = content.content_id

            logger.info(f"[{content_id_str}] {'DRY RUN: ' if dry_run else ''}Fixing language mismatch...")

            if dry_run:
                # Count what would be deleted
                from sqlalchemy import func

                # Count theme_classifications
                segment_ids = session.query(EmbeddingSegment.id).filter_by(content_id=content_id_int)
                tc_count = session.query(func.count()).filter(
                    ThemeClassification.segment_id.in_(segment_ids)
                ).scalar() or 0
                results['deleted']['theme_classifications'] = tc_count

                # Count alternative_transcriptions
                at_count = session.query(func.count()).filter(
                    AlternativeTranscription.segment_id.in_(segment_ids)
                ).scalar() or 0
                results['deleted']['alternative_transcriptions'] = at_count

                # Count embedding_segments
                es_count = session.query(func.count(EmbeddingSegment.id)).filter_by(
                    content_id=content_id_int
                ).scalar() or 0
                results['deleted']['embedding_segments'] = es_count

                # Count sentences
                from src.database.models import Sentence
                sent_count = session.query(func.count(Sentence.id)).filter_by(
                    content_id=content_id_int
                ).scalar() or 0
                results['deleted']['sentences'] = sent_count

                # Count speaker_transcriptions
                st_count = session.query(func.count(SpeakerTranscription.id)).filter_by(
                    content_id=content_id_int
                ).scalar() or 0
                results['deleted']['speaker_transcriptions'] = st_count

                # Count S3 files (estimate based on chunks)
                chunk_count = session.query(func.count(ContentChunk.id)).filter_by(
                    content_id=content_id_int
                ).scalar() or 0
                results['deleted']['s3_files'] = chunk_count  # One transcript per chunk

                results['reset_flags'] = ['is_transcribed', 'is_stitched', 'is_embedded']
                results['task_created'] = f'{chunk_count} transcribe tasks (high priority)'
                results['success'] = True

                logger.info(f"[{content_id_str}] DRY RUN would delete: "
                           f"{tc_count} classifications, {at_count} alt_trans, "
                           f"{es_count} segments, {sent_count} sentences, "
                           f"{st_count} speaker_trans, ~{chunk_count} S3 files")

            else:
                # Actually perform the fix
                from sqlalchemy import func
                from src.database.models import Sentence

                # 1. Delete theme_classifications for this content's segments
                segment_ids_subq = session.query(EmbeddingSegment.id).filter_by(content_id=content_id_int)
                tc_deleted = session.query(ThemeClassification).filter(
                    ThemeClassification.segment_id.in_(segment_ids_subq)
                ).delete(synchronize_session=False)
                results['deleted']['theme_classifications'] = tc_deleted
                logger.info(f"[{content_id_str}] Deleted {tc_deleted} theme_classifications")

                # 2. Delete alternative_transcriptions for this content's segments
                at_deleted = session.query(AlternativeTranscription).filter(
                    AlternativeTranscription.segment_id.in_(segment_ids_subq)
                ).delete(synchronize_session=False)
                results['deleted']['alternative_transcriptions'] = at_deleted
                logger.info(f"[{content_id_str}] Deleted {at_deleted} alternative_transcriptions")

                # 3. Delete embedding_segments
                es_deleted = session.query(EmbeddingSegment).filter_by(
                    content_id=content_id_int
                ).delete(synchronize_session=False)
                results['deleted']['embedding_segments'] = es_deleted
                logger.info(f"[{content_id_str}] Deleted {es_deleted} embedding_segments")

                # 4. Delete sentences
                sent_deleted = session.query(Sentence).filter_by(
                    content_id=content_id_int
                ).delete(synchronize_session=False)
                results['deleted']['sentences'] = sent_deleted
                logger.info(f"[{content_id_str}] Deleted {sent_deleted} sentences")

                # 5. Delete speaker_transcriptions
                st_deleted = session.query(SpeakerTranscription).filter_by(
                    content_id=content_id_int
                ).delete(synchronize_session=False)
                results['deleted']['speaker_transcriptions'] = st_deleted
                logger.info(f"[{content_id_str}] Deleted {st_deleted} speaker_transcriptions")

                # 6. Delete S3 transcript files
                s3_deleted = 0
                prefix = f"content/{content_id_str}/chunks/"
                try:
                    objects = self.s3_storage.list_files(prefix)
                    for obj_key in objects:
                        if 'transcript' in obj_key.lower():
                            if self.s3_storage.delete_file(obj_key):
                                s3_deleted += 1
                                logger.debug(f"[{content_id_str}] Deleted S3: {obj_key}")
                except Exception as e:
                    logger.warning(f"[{content_id_str}] Error deleting S3 transcripts: {e}")
                results['deleted']['s3_files'] = s3_deleted
                logger.info(f"[{content_id_str}] Deleted {s3_deleted} S3 transcript files")

                # 7. Reset content_chunks transcription status
                chunks_reset = session.query(ContentChunk).filter_by(
                    content_id=content_id_int
                ).update({
                    'transcription_status': 'pending',
                    'transcription_worker_id': None,
                    'transcription_attempts': 0,
                    'transcription_completed_at': None,
                    'transcription_error': None,
                    'transcribed_with': None
                }, synchronize_session=False)
                logger.info(f"[{content_id_str}] Reset {chunks_reset} chunks to pending")

                # 8. Reset content flags
                content.is_transcribed = False
                content.is_stitched = False
                content.is_embedded = False
                content.stitch_version = None
                content.last_updated = dt.now()
                results['reset_flags'] = ['is_transcribed', 'is_stitched', 'is_embedded']

                # 9. Delete ALL existing transcribe/stitch/embed tasks (unique constraint includes completed)
                deleted_tasks = session.query(TaskQueue).filter(
                    TaskQueue.content_id == content_id_str,
                    TaskQueue.task_type.in_(['transcribe', 'stitch', 'embed'])
                ).delete(synchronize_session=False)
                logger.info(f"[{content_id_str}] Deleted {deleted_tasks} existing tasks (all statuses)")

                # 10. Create transcribe task for each chunk with proper priority
                from src.utils.priority import calculate_priority_by_date
                project_priority = 1
                if content.projects:
                    for proj_name in content.projects:
                        if proj_name in self.config.get('active_projects', {}):
                            proj_config = self.config['active_projects'][proj_name]
                            project_priority = max(project_priority, proj_config.get('priority', 1))
                priority = calculate_priority_by_date(content.publish_date, project_priority)

                # Get all chunks for this content
                chunks = session.query(ContentChunk).filter_by(
                    content_id=content_id_int
                ).order_by(ContentChunk.chunk_index).all()

                tasks_created = 0
                for chunk in chunks:
                    new_task = TaskQueue(
                        content_id=content_id_str,
                        task_type='transcribe',
                        status='pending',
                        priority=priority,
                        input_data={
                            'chunk_index': chunk.chunk_index,
                            'reason': 'fix_language_mismatch',
                            'original_language': content.main_language,
                            'rewrite': True
                        }
                    )
                    session.add(new_task)
                    tasks_created += 1

                results['task_created'] = f'{tasks_created} transcribe tasks (priority={priority})'

                # Commit all changes
                session.commit()
                results['success'] = True

                logger.info(f"[{content_id_str}] ✓ Language mismatch fix complete. "
                           f"Created transcribe task with priority {priority}")

        except Exception as e:
            results['errors'].append(str(e))
            logger.error(f"[{content.content_id}] Error fixing language mismatch: {e}", exc_info=True)
            session.rollback()

        return results

    async def unblock_stuck_content(self, project: str = None, limit: int = 100,
                                   dry_run: bool = False, verbose: bool = False, stuck_state_filter: str = None,
                                   skip_quality_check: bool = True, fix_issues: bool = False) -> Dict[str, Any]:
        """Find and unblock stuck content by auditing items without active tasks.
        
        Finds content that:
        1. Is not compressed (not in final state)
        2. Is not blocked
        3. Has no pending/processing tasks
        4. Optionally matches a specific stuck state filter
        
        Then runs audit on each to fix state and create next task.
        
        Args:
            stuck_state_filter: Filter for specific stuck state (e.g. 'transcribed_not_diarized')
        """
        logger.info(f"Starting unblock for stuck content (project={project}, limit={limit}, stuck_state_filter={stuck_state_filter}, dry_run={dry_run}, fix_issues={fix_issues})")
        
        results = {
            'unblock_timestamp': dt.now().isoformat(),
            'project': project,
            'limit': limit,
            'stuck_state_filter': stuck_state_filter,
            'dry_run': dry_run,
            'fix_issues': fix_issues,
            'total_incomplete': 0,
            'total_with_tasks': 0,
            'total_stuck': 0,
            'total_unblocked': 0,
            'content_audited': [],
            'errors': [],
            'summary': {
                'tasks_created': defaultdict(int),
                'stuck_states': defaultdict(int),
                'errors_by_type': defaultdict(int),
                'flags_updated': defaultdict(int),
                'tasks_deleted': defaultdict(int),
                'corrupted_files_fixed': 0
            }
        }
        
        try:
            with get_session() as session:
                # Step 1: Find all incomplete content
                query = session.query(Content).filter(
                    Content.is_compressed == False,
                    (Content.blocked_download == False) | (Content.blocked_download == None),
                    (Content.is_short == False) | (Content.is_short == None),  # Exclude short content
                    (Content.is_duplicate == False) | (Content.is_duplicate == None)  # Exclude duplicate content
                )
                
                # Filter by project if specified
                if project:
                    # Projects is now stored as an array
                    from sqlalchemy import func
                    query = query.filter(
                        func.array_position(Content.projects, project) != None
                    )
                    
                    # Apply date filter if project has date range configured
                    if project in self.config.get('active_projects', {}):
                        project_config = self.config['active_projects'][project]
                        if 'start_date' in project_config and project_config['start_date']:
                            start_date = dt.fromisoformat(project_config['start_date'])
                            query = query.filter(Content.publish_date >= start_date)
                            logger.info(f"Applying start date filter for {project}: {start_date}")
                        if 'end_date' in project_config and project_config['end_date']:
                            end_date = dt.fromisoformat(project_config['end_date'])
                            query = query.filter(Content.publish_date <= end_date)
                            logger.info(f"Applying end date filter for {project}: {end_date}")
                else:
                    # When no specific project is provided, still respect enabled project time bounds
                    # Build OR conditions for each enabled project's date range
                    from sqlalchemy import or_, and_
                    
                    project_filters = []
                    active_projects = self.config.get('active_projects', {})
                    
                    for proj_name, proj_config in active_projects.items():
                        if not proj_config.get('enabled', True):
                            continue

                        # Build project match condition (projects is now an array)
                        from sqlalchemy import func
                        project_match = func.array_position(Content.projects, proj_name) != None
                        
                        # Build date range condition for this project
                        date_conditions = []
                        if 'start_date' in proj_config and proj_config['start_date']:
                            start_date = dt.fromisoformat(proj_config['start_date'])
                            date_conditions.append(Content.publish_date >= start_date)
                        if 'end_date' in proj_config and proj_config['end_date']:
                            end_date = dt.fromisoformat(proj_config['end_date'])
                            date_conditions.append(Content.publish_date <= end_date)
                        
                        # Combine project match with its date range
                        if date_conditions:
                            project_filter = and_(project_match, *date_conditions)
                        else:
                            project_filter = project_match
                        
                        project_filters.append(project_filter)
                    
                    # Apply OR of all enabled project filters
                    if project_filters:
                        query = query.filter(or_(*project_filters))
                        logger.info(f"Applying date bounds for {len(project_filters)} enabled projects")
                
                incomplete_content = query.all()
                results['total_incomplete'] = len(incomplete_content)
                logger.info(f"Found {len(incomplete_content)} incomplete content items")
                
                if not incomplete_content:
                    logger.info("No incomplete content found")
                    return results
                
                # Step 2: Get all content IDs with pending/processing tasks
                active_tasks = session.query(TaskQueue.content_id).filter(
                    TaskQueue.status.in_(['pending', 'processing'])
                ).distinct().all()
                
                active_content_ids = {task[0] for task in active_tasks}
                results['total_with_tasks'] = len(active_content_ids)
                logger.info(f"Found {len(active_content_ids)} content items with active tasks")
                
                # Step 3: Find stuck content (incomplete with no active tasks)
                stuck_content = []
                for content in incomplete_content:
                    if content.content_id not in active_content_ids:
                        # Determine stuck state for reporting
                        state = self._determine_stuck_state(content)
                        
                        # Apply stuck state filter if specified
                        if stuck_state_filter and state != stuck_state_filter:
                            if verbose:
                                logger.debug(f"Skipping {content.content_id} - state '{state}' doesn't match filter '{stuck_state_filter}'")
                            continue
                        
                        stuck_content.append((content.content_id, state))
                        results['summary']['stuck_states'][state] += 1
                    else:
                        if verbose:
                            logger.debug(f"Skipping {content.content_id} - has active tasks")
                
                # Sort stuck content to prioritize partially processed items over not_downloaded
                def priority_key(item):
                    content_id, state = item
                    # Higher priority (lower number) for partially processed content
                    priority_map = {
                        'diarized_not_transcribed': 1,  # Almost done, just needs transcription
                        'both_not_stitched': 2,          # Transcribed and diarized, needs stitching
                        'transcribed_not_diarized': 3,   # Has transcription, needs diarization
                        'segmented_not_compressed': 4,   # Almost complete, needs compression
                        'converted_not_processed': 5,    # Has audio, needs processing
                        'downloaded_not_converted': 6,   # Has source, needs conversion
                        'stitched_not_segmented': 7,     # Needs segmentation
                        'not_downloaded': 8,              # Lowest priority - hasn't started
                        'unknown_state': 9                # Unknown states last
                    }
                    return priority_map.get(state, 9)
                
                stuck_content.sort(key=priority_key)
                logger.info(f"Sorted stuck content by processing priority")
                
                results['total_stuck'] = len(stuck_content)
                if stuck_state_filter:
                    logger.info(f"Found {len(stuck_content)} stuck content items matching state '{stuck_state_filter}'")
                else:
                    logger.info(f"Found {len(stuck_content)} stuck content items")
                
                if not stuck_content:
                    logger.info("No stuck content found - all incomplete content has active tasks")
                    return results
                
                # Apply limit if specified
                if limit > 0:
                    stuck_content = stuck_content[:limit]
                    logger.info(f"Processing first {limit} stuck content items")
                
                # Step 4: Audit each stuck content item
                logger.info(f"{'DRY RUN: Would audit' if dry_run else 'Auditing'} {len(stuck_content)} stuck content items...")
                
                with tqdm(total=len(stuck_content), desc="Unblocking content", unit="item") as pbar:
                    for content_id, stuck_state in stuck_content:
                        pbar.set_description(f"Auditing {content_id[:16]}...")
                        
                        try:
                            # Run audit to fix state and create appropriate task
                            audit_result = await self.audit_content(
                                content_id,
                                dry_run=dry_run,
                                verbose=verbose,
                                skip_quality_check=skip_quality_check,
                                fix_issues=fix_issues
                            )
                            
                            # Track results
                            if audit_result['status'] == 'success':
                                # Check if tasks were created
                                if audit_result.get('evaluation_results', {}).get('tasks_created'):
                                    results['total_unblocked'] += 1
                                    for task_type in audit_result['evaluation_results']['tasks_created']:
                                        results['summary']['tasks_created'][task_type] += 1
                                
                                # Track flags updated
                                if audit_result.get('evaluation_results', {}).get('flags_updated'):
                                    for flag in audit_result['evaluation_results']['flags_updated']:
                                        results['summary']['flags_updated'][flag] += 1
                                
                                # Track tasks deleted
                                if audit_result.get('tasks_deleted'):
                                    for task_str in audit_result['tasks_deleted']:
                                        # Extract task type from string like "transcribe (ID: 123)"
                                        task_type = task_str.split(' (')[0] if ' (' in task_str else task_str
                                        results['summary']['tasks_deleted'][task_type] += 1

                                # Track corrupted files fixed
                                if audit_result.get('corrupted_files_deleted'):
                                    results['summary']['corrupted_files_fixed'] += 1

                                results['content_audited'].append({
                                    'content_id': content_id,
                                    'stuck_state': stuck_state,
                                    'actions_taken': audit_result.get('actions_taken', []),
                                    'tasks_created': audit_result.get('evaluation_results', {}).get('tasks_created', []),
                                    'flags_updated': audit_result.get('evaluation_results', {}).get('flags_updated', []),
                                    'tasks_deleted': audit_result.get('tasks_deleted', []),
                                    'corrupted_files_deleted': audit_result.get('corrupted_files_deleted', [])
                                })
                            else:
                                results['errors'].append(f"{content_id}: {audit_result.get('errors', ['Unknown error'])}")
                                results['summary']['errors_by_type']['audit_failed'] += 1
                            
                        except Exception as e:
                            error_msg = f"Failed to audit {content_id}: {str(e)}"
                            logger.error(error_msg)
                            results['errors'].append(error_msg)
                            results['summary']['errors_by_type'][type(e).__name__] += 1
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            "Unblocked": results['total_unblocked'],
                            "Errors": len(results['errors'])
                        })
                
                return results
                
        except Exception as e:
            error_msg = f"Unblock operation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            results['errors'].append(error_msg)
            return results
    
    def _determine_stuck_state(self, content: Content) -> str:
        """Determine the stuck state of content for reporting."""
        if content.is_stitched and not content.is_embedded:
            return "stitched_not_segmented"
        elif content.is_diarized and not content.is_transcribed:
            return "diarized_not_transcribed"
        elif content.is_transcribed and not content.is_diarized:
            return "transcribed_not_diarized"
        elif content.is_transcribed and content.is_diarized and not content.is_stitched:
            return "both_not_stitched"
        elif content.is_embedded and not content.is_compressed:
            return "segmented_not_compressed"
        elif content.is_converted and not content.is_transcribed and not content.is_diarized:
            return "converted_not_processed"
        elif content.is_downloaded and not content.is_converted:
            return "downloaded_not_converted"
        elif not content.is_downloaded:
            return "not_downloaded"
        else:
            return "unknown_state"

    async def audit_multiple_from_queue(self, task_type: str, task_status: str = 'failed',
                                        limit: int = 10, dry_run: bool = False,
                                        verbose: bool = False, skip_quality_check: bool = False) -> Dict[str, Any]:
        """Audit multiple content items based on task queue criteria."""
        logger.info(f"Starting batch audit for {limit} content items with {task_status} {task_type} tasks")
        
        batch_results = {
            'task_type': task_type,
            'task_status': task_status,
            'limit': limit,
            'dry_run': dry_run,
            'total_found': 0,
            'total_audited': 0,
            'total_actions': 0,
            'total_tasks_deleted': 0,
            'results': [],
            'errors': [],
            'summary': {
                'content_already_correct': 0,
                'content_with_actions': 0,
                'content_with_errors': 0,
                'content_with_deleted_tasks': 0
            }
        }
        
        try:
            with get_session() as session:
                # Query tasks of specified type and status
                # Order by priority (desc) and created_at (asc) to match task assignment logic
                query = session.query(TaskQueue).filter(
                    TaskQueue.task_type == task_type,
                    TaskQueue.status == task_status
                ).order_by(TaskQueue.priority.desc(), TaskQueue.created_at.asc())
                
                if limit > 0:
                    query = query.limit(limit * 2)  # Get more tasks in case some content IDs are duplicates
                
                tasks = query.all()
                batch_results['total_found'] = len(tasks)
                
                if not tasks:
                    logger.warning(f"No {task_status} {task_type} tasks found in queue")
                    return batch_results
                
                # Get unique content IDs
                content_ids = list(dict.fromkeys([task.content_id for task in tasks]))  # Preserve order, remove duplicates
                
                if limit > 0:
                    content_ids = content_ids[:limit]
                
                logger.info(f"Found {len(content_ids)} unique content items to audit")
                
                # Audit each content item with progress bar
                with tqdm(total=len(content_ids), desc="Auditing content", unit="item") as pbar:
                    for i, content_id in enumerate(content_ids, 1):
                        pbar.set_description(f"Auditing {content_id[:16]}...")
                        
                        try:
                            result = await self.audit_content(content_id, dry_run, verbose, skip_quality_check)
                            batch_results['results'].append(result)
                            batch_results['total_audited'] += 1
                            
                            # Count actions
                            action_count = len([a for a in result.get('actions_taken', []) 
                                              if not a.startswith('No changes needed')])
                            batch_results['total_actions'] += action_count
                            
                            # Count deleted tasks
                            deleted_count = len(result.get('tasks_deleted', []))
                            batch_results['total_tasks_deleted'] += deleted_count
                            
                            # Update summary
                            if result['status'] == 'error':
                                batch_results['summary']['content_with_errors'] += 1
                                pbar.set_postfix({"Errors": batch_results['summary']['content_with_errors']})
                            elif action_count > 0 or deleted_count > 0:
                                batch_results['summary']['content_with_actions'] += 1
                                if deleted_count > 0:
                                    batch_results['summary']['content_with_deleted_tasks'] += 1
                                pbar.set_postfix({"Actions": batch_results['summary']['content_with_actions']})
                            else:
                                batch_results['summary']['content_already_correct'] += 1
                                pbar.set_postfix({"OK": batch_results['summary']['content_already_correct']})
                                
                        except Exception as e:
                            error_msg = f"Failed to audit {content_id}: {str(e)}"
                            logger.error(error_msg)
                            batch_results['errors'].append(error_msg)
                            batch_results['results'].append({
                                'content_id': content_id,
                                'status': 'error',
                                'errors': [error_msg]
                            })
                            batch_results['summary']['content_with_errors'] += 1
                            pbar.set_postfix({"Errors": batch_results['summary']['content_with_errors']})
                        
                        pbar.update(1)
                
                return batch_results
                
        except Exception as e:
            error_msg = f"Batch audit failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            batch_results['errors'].append(error_msg)
            return batch_results

    async def audit_quality_by_project(self, project: str, limit: int = 0,
                                       dry_run: bool = False, verbose: bool = False,
                                       fix_issues: bool = False, language: str = None) -> Dict[str, Any]:
        """Audit all transcribed content for a project for quality issues.

        Unlike unblock mode, this targets ALL content with transcripts (including completed)
        to catch language mismatch and other quality issues.

        Args:
            project: Project to audit (required)
            limit: Max items to process (0 = all)
            dry_run: If True, only report what would be done
            fix_issues: If True, fix detected issues (e.g., language mismatch)
            language: Filter by main_language (e.g., 'fr' for French)
        """
        logger.info(f"Starting quality audit for project {project} (limit={limit}, language={language}, fix_issues={fix_issues}, dry_run={dry_run})")

        results = {
            'audit_timestamp': dt.now().isoformat(),
            'project': project,
            'language': language,
            'limit': limit,
            'dry_run': dry_run,
            'fix_issues': fix_issues,
            'total_content': 0,
            'total_audited': 0,
            'total_with_issues': 0,
            'total_fixed': 0,
            'total_skipped': 0,
            'content_results': [],
            'errors': [],
            'summary': {
                'issues_by_type': defaultdict(int),
                'fixes_by_type': defaultdict(int),
                'errors_by_type': defaultdict(int)
            }
        }

        try:
            with get_session() as session:
                from sqlalchemy import func

                # Query content with transcripts (has embedding segments)
                # Join with embedding_segments to ensure content has been processed
                query = session.query(Content).join(
                    EmbeddingSegment,
                    Content.id == EmbeddingSegment.content_id
                ).filter(
                    func.array_position(Content.projects, project) != None
                ).distinct()

                # Filter by language if specified
                if language:
                    query = query.filter(Content.main_language == language)

                # Get content items
                content_items = query.all()
                results['total_content'] = len(content_items)
                logger.info(f"Found {len(content_items)} content items with transcripts")

                if not content_items:
                    logger.info("No content found matching criteria")
                    return results

                # Apply limit if specified
                if limit > 0:
                    content_items = content_items[:limit]
                    logger.info(f"Processing first {limit} items")

                # Get content IDs with pending transcribe tasks to skip
                pending_transcribe = session.query(TaskQueue.content_id).filter(
                    TaskQueue.task_type == 'transcribe',
                    TaskQueue.status.in_(['pending', 'processing'])
                ).distinct().all()
                pending_ids = {t[0] for t in pending_transcribe}
                logger.info(f"Found {len(pending_ids)} content IDs with pending transcribe tasks")

                # Audit each content item
                with tqdm(total=len(content_items), desc="Auditing quality", unit="item") as pbar:
                    for content in content_items:
                        pbar.set_description(f"Auditing {content.content_id[:16]}...")

                        # Skip if already has pending transcribe task
                        if content.content_id in pending_ids:
                            results['total_skipped'] += 1
                            pbar.update(1)
                            continue

                        try:
                            # Run audit with quality check only (no state reconciliation)
                            # This prevents creating stitch/embed tasks for unrelated issues
                            audit_result = await self.audit_content(
                                content.content_id,
                                dry_run=dry_run,
                                verbose=verbose,
                                skip_quality_check=False,  # Always run quality check
                                fix_issues=fix_issues,
                                skip_state_reconciliation=True  # Only check quality, don't create other tasks
                            )

                            results['total_audited'] += 1

                            # Track quality issues
                            if audit_result.get('transcript_quality'):
                                quality = audit_result['transcript_quality']
                                if quality.get('issues'):
                                    results['total_with_issues'] += 1
                                    for issue in quality['issues']:
                                        results['summary']['issues_by_type'][issue.get('type', 'unknown')] += 1

                            # Track fixes
                            if audit_result.get('issues_fixed'):
                                for fix in audit_result['issues_fixed']:
                                    if fix.get('fix_result', {}).get('success'):
                                        results['total_fixed'] += 1
                                        results['summary']['fixes_by_type'][fix.get('type', 'unknown')] += 1

                            # Store summary for verbose output
                            if verbose or audit_result.get('issues_fixed'):
                                results['content_results'].append({
                                    'content_id': content.content_id,
                                    'channel': content.channel_name,
                                    'issues': [i.get('type') for i in audit_result.get('transcript_quality', {}).get('issues', [])],
                                    'fixed': [f.get('type') for f in audit_result.get('issues_fixed', [])]
                                })

                        except Exception as e:
                            error_msg = f"Failed to audit {content.content_id}: {str(e)}"
                            logger.error(error_msg)
                            results['errors'].append(error_msg)
                            results['summary']['errors_by_type'][type(e).__name__] += 1

                        pbar.update(1)
                        pbar.set_postfix({
                            "Issues": results['total_with_issues'],
                            "Fixed": results['total_fixed'],
                            "Skipped": results['total_skipped']
                        })

                return results

        except Exception as e:
            error_msg = f"Quality audit failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            results['errors'].append(error_msg)
            return results


def print_quality_audit_report(results: Dict[str, Any], verbose: bool = False):
    """Print a formatted quality audit report."""
    dry_run = results['dry_run']
    project = results.get('project', 'All')
    language = results.get('language')

    print("=" * 80)
    print(f"QUALITY AUDIT REPORT")
    filter_text = f"Project: {project}"
    if language:
        filter_text += f" | Language: {language}"
    print(f"{'DRY RUN' if dry_run else 'LIVE RUN'} - {filter_text}")
    print("=" * 80)

    # Summary stats
    print(f"\n📊 SUMMARY:")
    print(f"   Total content found: {results['total_content']}")
    print(f"   Total audited: {results['total_audited']}")
    print(f"   Skipped (pending tasks): {results['total_skipped']}")
    print(f"   With quality issues: {results['total_with_issues']}")
    print(f"   Fixed: {results['total_fixed']}")

    # Issues breakdown
    if results['summary']['issues_by_type']:
        print(f"\n⚠️  ISSUES FOUND:")
        for issue_type, count in sorted(results['summary']['issues_by_type'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {issue_type}: {count}")

    # Fixes breakdown
    if results['summary']['fixes_by_type']:
        print(f"\n✅ FIXES APPLIED:")
        for fix_type, count in sorted(results['summary']['fixes_by_type'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {fix_type}: {count}")

    # Errors
    if results['errors']:
        print(f"\n❌ ERRORS ({len(results['errors'])}):")
        for error in results['errors'][:5]:
            print(f"   {error[:100]}")
        if len(results['errors']) > 5:
            print(f"   ... and {len(results['errors']) - 5} more errors")

    print("\n" + "=" * 80)
    if dry_run:
        print(f"📝 DRY RUN COMPLETE - No actual changes were made")
        if results['total_with_issues'] > 0:
            print(f"   Would fix {results['total_with_issues']} content items with issues")
    else:
        print(f"✅ AUDIT COMPLETE")
        if results['total_fixed'] > 0:
            print(f"   Fixed {results['total_fixed']} content items")
    print("=" * 80)


def print_unblock_report(results: Dict[str, Any], verbose: bool = False):
    """Print a formatted unblock operation report."""
    dry_run = results['dry_run']
    project = results.get('project', 'All')
    stuck_state_filter = results.get('stuck_state_filter')
    
    print("=" * 80)
    print(f"UNBLOCK STUCK CONTENT REPORT")
    filter_text = f"Project: {project}"
    if stuck_state_filter:
        filter_text += f" | State: {stuck_state_filter}"
    print(f"{'DRY RUN' if dry_run else 'LIVE RUN'} - {filter_text}")
    print("=" * 80)
    
    # Summary stats
    print(f"\n📊 SUMMARY:")
    print(f"   Total incomplete content: {results['total_incomplete']}")
    print(f"   Content with active tasks: {results['total_with_tasks']}")
    print(f"   Stuck content found: {results['total_stuck']}")
    print(f"   Content unblocked: {results['total_unblocked']}")
    
    # Stuck states breakdown
    if results['summary']['stuck_states']:
        print(f"\n📋 STUCK STATES FOUND:")
        state_names = {
            'stitched_not_segmented': 'Stitched but not segmented',
            'diarized_not_transcribed': 'Diarized but not transcribed',
            'transcribed_not_diarized': 'Transcribed but not diarized',
            'both_not_stitched': 'Transcribed & diarized but not stitched',
            'segmented_not_compressed': 'Segmented but not compressed',
            'converted_not_processed': 'Converted but not transcribed/diarized',
            'downloaded_not_converted': 'Downloaded but not converted',
            'not_downloaded': 'Not downloaded',
            'unknown_state': 'Unknown state'
        }
        for state, count in sorted(results['summary']['stuck_states'].items(), key=lambda x: x[1], reverse=True):
            state_name = state_names.get(state, state)
            print(f"   {state_name}: {count}")
    
    # Tasks created breakdown - ALWAYS show this section
    print(f"\n✅ TASKS CREATED SUMMARY:")
    if results['summary']['tasks_created']:
        total_tasks = sum(results['summary']['tasks_created'].values())
        print(f"   Total tasks created: {total_tasks}")
        print(f"\n   Breakdown by type:")
        for task_type, count in sorted(results['summary']['tasks_created'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_tasks * 100) if total_tasks > 0 else 0
            print(f"   {task_type}: {count} ({percentage:.1f}%)")
    else:
        print(f"   No new tasks created")
    
    # Flags updated summary
    if results['summary'].get('flags_updated'):
        print(f"\n🔧 FLAGS UPDATED:")
        total_flag_updates = sum(results['summary']['flags_updated'].values())
        print(f"   Total flag updates: {total_flag_updates}")
        for flag, count in sorted(results['summary']['flags_updated'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {flag}: {count}")
    
    # Tasks deleted summary
    if results['summary'].get('tasks_deleted'):
        print(f"\n🗑️  INAPPROPRIATE TASKS DELETED:")
        total_deleted = sum(results['summary']['tasks_deleted'].values())
        print(f"   Total tasks deleted: {total_deleted}")
        for task_type, count in sorted(results['summary']['tasks_deleted'].items(), key=lambda x: x[1], reverse=True):
            print(f"   {task_type}: {count}")

    # Corrupted files fixed
    if results['summary'].get('corrupted_files_fixed', 0) > 0:
        print(f"\n🔧 CORRUPTED SOURCE FILES FIXED:")
        print(f"   Content items with corrupted audio: {results['summary']['corrupted_files_fixed']}")
        print(f"   Action: Reset download flag and deleted corrupted files for re-download")

    # Individual results (if verbose or if there were actions)
    if verbose and results['content_audited']:
        print(f"\n📋 CONTENT DETAILS:")
        for item in results['content_audited'][:20]:  # Show first 20
            content_id = item['content_id']
            stuck_state = item['stuck_state']
            tasks_created = item['tasks_created']
            
            if tasks_created:
                print(f"   {content_id[:16]}... ({stuck_state})")
                print(f"      → Created: {', '.join(tasks_created)}")
        
        if len(results['content_audited']) > 20:
            print(f"   ... and {len(results['content_audited']) - 20} more items")
    
    # Errors
    if results['errors']:
        print(f"\n⚠️  ERRORS ({len(results['errors'])}):")
        for error in results['errors'][:10]:  # Show first 10 errors
            print(f"   • {error}")
        if len(results['errors']) > 10:
            print(f"   ... and {len(results['errors']) - 10} more errors")
        
        if results['summary']['errors_by_type']:
            print(f"\n   Error types:")
            for error_type, count in results['summary']['errors_by_type'].items():
                print(f"      {error_type}: {count}")
    
    # Final status and action summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY:")
    
    if dry_run:
        print(f"📝 DRY RUN COMPLETE - No actual changes were made")
        if results['summary']['tasks_created']:
            total_tasks = sum(results['summary']['tasks_created'].values())
            print(f"   Would create {total_tasks} new tasks")
        if results['summary'].get('flags_updated'):
            total_flags = sum(results['summary']['flags_updated'].values())
            print(f"   Would update {total_flags} database flags")
        if results['summary'].get('tasks_deleted'):
            total_deleted = sum(results['summary']['tasks_deleted'].values())
            print(f"   Would delete {total_deleted} inappropriate tasks")
    else:
        if results['total_unblocked'] > 0:
            print(f"🎯 SUCCESSFULLY UNBLOCKED {results['total_unblocked']} content items")
            if results['summary']['tasks_created']:
                total_tasks = sum(results['summary']['tasks_created'].values())
                print(f"   Created {total_tasks} new tasks")
            if results['summary'].get('flags_updated'):
                total_flags = sum(results['summary']['flags_updated'].values())
                print(f"   Updated {total_flags} database flags")
            if results['summary'].get('tasks_deleted'):
                total_deleted = sum(results['summary']['tasks_deleted'].values())
                print(f"   Deleted {total_deleted} inappropriate tasks")
        elif results['total_stuck'] > 0:
            print(f"⚠️  Found {results['total_stuck']} stuck items but none were unblocked")
        else:
            print(f"✅ No stuck content found - all incomplete content has active tasks")
    
    if results['errors']:
        print(f"\n⚠️  Encountered {len(results['errors'])} errors during processing")
    
    print("=" * 80)

def print_batch_audit_report(batch_results: Dict[str, Any], verbose: bool = False):
    """Print a formatted batch audit report."""
    task_type = batch_results['task_type']
    task_status = batch_results['task_status']
    dry_run = batch_results['dry_run']
    
    print("=" * 80)
    print(f"BATCH AUDIT REPORT - {task_status.upper()} {task_type.upper()} TASKS")
    print(f"{'DRY RUN' if dry_run else 'LIVE RUN'}")
    print("=" * 80)
    
    # Summary stats
    print(f"\n📊 SUMMARY:")
    print(f"   Task type: {task_type}")
    print(f"   Task status: {task_status}")
    print(f"   Tasks found: {batch_results['total_found']}")
    print(f"   Content audited: {batch_results['total_audited']}")
    print(f"   Total actions: {batch_results['total_actions']}")
    print(f"   Tasks deleted: {batch_results['total_tasks_deleted']}")
    
    # Results breakdown
    summary = batch_results['summary']
    print(f"\n📈 RESULTS BREAKDOWN:")
    print(f"   ✅ Already correct: {summary['content_already_correct']}")
    print(f"   🔧 Needed fixes: {summary['content_with_actions']}")
    print(f"   🗑️  Had inappropriate tasks: {summary['content_with_deleted_tasks']}")
    print(f"   ❌ Had errors: {summary['content_with_errors']}")
    
    # Individual results (if verbose or if there are actions/errors)
    results = batch_results['results']
    
    if verbose or any(len(r.get('actions_taken', [])) > 1 or r.get('status') == 'error' or r.get('tasks_deleted') for r in results):
        print(f"\n📋 INDIVIDUAL RESULTS:")
        
        for result in results:
            content_id = result['content_id']
            status = result['status']
            actions = result.get('actions_taken', [])
            errors = result.get('errors', [])
            deleted_tasks = result.get('tasks_deleted', [])
            
            # Skip "no changes needed" content unless verbose or has deleted tasks
            if not verbose and len(actions) == 1 and actions[0].startswith('No changes needed') and not deleted_tasks:
                continue
            
            # Status icon
            if status == 'error':
                icon = "❌"
            elif len(actions) > 1 or (actions and not actions[0].startswith('No changes needed')) or deleted_tasks:
                icon = "🔧"
            else:
                icon = "✅"
            
            print(f"\n   {icon} {content_id[:16]}...")
            
            # Show deleted tasks first
            if deleted_tasks:
                task_types = [t.split(' (')[0] for t in deleted_tasks]
                print(f"      • {'Would delete' if batch_results['dry_run'] else 'Deleted'} inappropriate tasks: {', '.join(task_types)}")
            
            # Show actions (skip "no changes needed" unless verbose)
            action_list = [a for a in actions if verbose or not a.startswith('No changes needed')]
            if action_list:
                for action in action_list:
                    if not action.startswith('Deleted') and not action.startswith('Would delete'):  # Don't duplicate deleted tasks
                        print(f"      • {action}")
            
            # Show errors
            if errors:
                for error in errors:
                    print(f"      ⚠️  {error}")
    
    # Batch errors
    if batch_results['errors']:
        print(f"\n⚠️  BATCH ERRORS:")
        for error in batch_results['errors']:
            print(f"   • {error}")
    
    print("=" * 80)

def print_audit_report(results: Dict[str, Any], verbose: bool = False):
    """Print a formatted audit report."""
    content_id = results['content_id']
    status = results['status']
    dry_run = results['dry_run']
    
    print("=" * 80)
    print(f"CONTENT AUDIT REPORT - {content_id}")
    print(f"{'DRY RUN' if dry_run else 'LIVE RUN'} - {results['audit_timestamp']}")
    print("=" * 80)
    
    if status == 'error':
        print("❌ AUDIT FAILED")
        for error in results['errors']:
            print(f"   Error: {error}")
        return
    
    # Before state
    before = results['before_state']
    print(f"\n📊 CONTENT INFO:")
    print(f"   ID: {before['content_id']}")
    print(f"   Platform: {before['platform']}")
    print(f"   Title: {before['title'] or 'N/A'}")
    print(f"   Duration: {before['duration']}s" if before['duration'] else "   Duration: Unknown")
    print(f"   Projects: {before['projects']}")
    if before['flags']['blocked_download']:
        print("   ⚠️  BLOCKED FOR DOWNLOAD")
    
    # Version information
    versions = before.get('versions', {})
    print(f"\n📋 VERSIONS:")
    print(f"   Stitch: {versions.get('stitch_version', 'Not set')}")
    print(f"   Segment: {versions.get('segment_version', 'Not set')}")
    print(f"   Diarization: {versions.get('diarization_method', 'Not set')}")
    
    # Current state flags
    print(f"\n🏁 PROCESSING FLAGS:")
    flags = before['flags']
    flag_icons = {
        'is_downloaded': '⬇️',
        'is_converted': '🔄', 
        'is_transcribed': '📝',
        'is_diarized': '🎭',
        'is_stitched': '🧵',
        'is_embedded': '🔍',
        'is_compressed': '📦'
    }
    
    for flag, value in flags.items():
        if flag == 'blocked_download':
            continue
        icon = flag_icons.get(flag, '📋')
        status_text = "✅" if value else "❌"
        print(f"   {icon} {flag.replace('is_', '').title()}: {status_text}")
    
    # Chunk info
    chunks = before['chunks']
    if chunks['total_chunks'] > 0:
        print(f"\n📦 CHUNKS ({chunks['total_chunks']} total):")
        print(f"   Extracted: {chunks['extracted_chunks']}/{chunks['total_chunks']}")
        print(f"   Transcribed: {chunks['transcribed_chunks']}/{chunks['total_chunks']}")
        
        # Show untranscribed chunks if there are any
        untranscribed_chunks = [c for c in chunks['chunk_details'] if c['transcription_status'] != 'completed']
        if untranscribed_chunks:
            print(f"   ⚠️  Untranscribed chunks: {[c['index'] for c in untranscribed_chunks]}")
        
        # Show unextracted chunks if there are any
        unextracted_chunks = [c for c in chunks['chunk_details'] if c['extraction_status'] != 'completed']
        if unextracted_chunks:
            print(f"   ⚠️  Unextracted chunks: {[c['index'] for c in unextracted_chunks]}")
        
        if verbose and chunks['chunk_details']:
            print("   Details:")
            for chunk in chunks['chunk_details'][:10]:  # Show first 10
                extract_status = "✅" if chunk['extraction_status'] == 'completed' else "❌"
                transcribe_status = "✅" if chunk['transcription_status'] == 'completed' else "❌"
                print(f"     Chunk {chunk['index']}: Extract {extract_status} Transcribe {transcribe_status}")
            if len(chunks['chunk_details']) > 10:
                print(f"     ... and {len(chunks['chunk_details']) - 10} more chunks")
    
    # Transcript Quality (if evaluated)
    if results.get('transcript_quality'):
        quality = results['transcript_quality']
        print(f"\n📝 TRANSCRIPT QUALITY:")

        # Severity indicator
        severity_icons = {
            'none': '✅',
            'low': '🟢',
            'medium': '🟡',
            'high': '🟠',
            'critical': '🔴',
            'unknown': '❓'
        }
        severity = quality.get('severity', 'unknown')
        icon = severity_icons.get(severity, '❓')
        print(f"   Overall Status: {icon} {severity.upper()}")

        # Show metrics
        metrics = quality.get('metrics', {})
        if metrics.get('total_issues', 0) > 0:
            print(f"   Total Issues: {metrics['total_issues']}")

            if verbose and 'by_type' in metrics:
                print("\n   Issues by Type:")
                for issue_type, count in metrics['by_type'].items():
                    print(f"      • {issue_type}: {count}")

        # Show issues
        issues = quality.get('issues', [])
        if issues:
            print("\n   Detected Issues:")
            for issue in issues:
                issue_type = issue.get('type', 'unknown')
                severity = issue.get('severity', 'unknown')
                message = issue.get('message', '')
                print(f"      [{severity.upper()}] {issue_type}: {message}")

                # Show samples if verbose
                if verbose and 'samples' in issue:
                    samples = issue['samples']
                    if samples:
                        print(f"         Samples: {json.dumps(samples[:2], indent=10)}")

        # Show recommendations
        recommendations = quality.get('recommendations', [])
        if recommendations:
            print("\n   Recommendations:")
            for rec in recommendations:
                print(f"      • {rec}")

    # Tasks
    pending = before['pending_tasks']
    if pending:
        print(f"\n⏳ PENDING TASKS ({len(pending)}):")
        for task in pending:
            status_icon = "⏳" if task['status'] == 'pending' else "🔄" if task['status'] == 'processing' else "❓"
            print(f"   {status_icon} {task['task_type']} (ID: {task['id']}, Priority: {task['priority']}, Status: {task['status']}, Created: {task['created_at']})")

    failed = before['recent_failed_tasks']
    if failed:
        print(f"\n❌ RECENT FAILED TASKS ({len(failed)}):")
        for task in failed:
            print(f"   {task['task_type']}: {task['error'][:100]}...")
    
    # Actions taken
    print(f"\n🔧 ACTIONS {'THAT WOULD BE TAKEN' if dry_run else 'TAKEN'}:")
    if results['actions_taken']:
        for action in results['actions_taken']:
            print(f"   • {action}")
    else:
        print("   • No actions needed")

    # Issues fixed (when --fix-issues is used)
    if results.get('issues_fixed'):
        print(f"\n🩹 ISSUES {'THAT WOULD BE' if dry_run else ''} FIXED:")
        for fix in results['issues_fixed']:
            fix_type = fix.get('type', 'unknown')
            fix_result = fix.get('fix_result', {})
            deleted = fix_result.get('deleted', {})
            print(f"   • {fix_type}:")
            if deleted:
                print(f"      Deleted: {deleted.get('embedding_segments', 0)} segments, "
                      f"{deleted.get('sentences', 0)} sentences, "
                      f"{deleted.get('theme_classifications', 0)} classifications")
            if fix_result.get('task_created'):
                print(f"      Task created: {fix_result['task_created']}")
            if fix_result.get('errors'):
                print(f"      Errors: {fix_result['errors']}")

    # Corrupted files deleted
    if results.get('corrupted_files_deleted'):
        print(f"\n🗑️  CORRUPTED FILES {'THAT WOULD BE' if dry_run else ''} DELETED:")
        for file_path in results['corrupted_files_deleted']:
            print(f"   • {file_path}")
    
    # Evaluation details
    if verbose and results['evaluation_results']:
        eval_results = results['evaluation_results']
        print(f"\n🔍 EVALUATION DETAILS:")
        if eval_results.get('flags_updated'):
            print(f"   Flags updated: {eval_results['flags_updated']}")
        if eval_results.get('tasks_created'):
            print(f"   Tasks created: {eval_results['tasks_created']}")
        if eval_results.get('errors'):
            print(f"   Errors: {eval_results['errors']}")
    
    # Changes summary
    if not dry_run and results['after_state']:
        after_flags = results['after_state']['flags']
        before_flags = before['flags']
        changes = []
        for flag, after_value in after_flags.items():
            before_value = before_flags.get(flag)
            if before_value != after_value:
                changes.append(f"{flag}: {before_value} → {after_value}")
        
        if changes:
            print(f"\n📈 CHANGES MADE:")
            for change in changes:
                print(f"   • {change}")
    
    # Errors
    if results['errors']:
        print(f"\n⚠️  ERRORS:")
        for error in results['errors']:
            print(f"   • {error}")
    
    print("=" * 80)

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Audit content state and create missing tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single content audit
  python audit_content.py --content abc123def456                    # Audit content (includes quality check)
  python audit_content.py --content abc123def456 --dry-run          # Show what would be done
  python audit_content.py --content abc123def456 --verbose          # Show detailed info including quality samples
  python audit_content.py --content abc123def456 --skip-quality-check  # Skip transcript quality check (faster)

  # Fix detected issues (e.g., language mismatch triggers re-transcription)
  python audit_content.py --content abc123def456 --fix-issues --dry-run  # Show what fixes would be applied
  python audit_content.py --content abc123def456 --fix-issues            # Actually fix detected issues

  # Batch audit from task queue
  python audit_content.py --task-type stitch --limit 5              # Audit 5 content with failed stitch tasks
  python audit_content.py --task-type transcribe --status pending   # Audit content with pending transcribe tasks
  python audit_content.py --task-type cleanup --limit 10 --dry-run  # Dry run for 10 content with failed cleanup

  # Unblock stuck content (quality check skipped by default for performance)
  python audit_content.py --unblock                                 # Find and unblock all stuck content
  python audit_content.py --unblock --project Big_Channels          # Unblock stuck content for specific project
  python audit_content.py --unblock --limit 50 --dry-run            # Dry run for first 50 stuck items
  python audit_content.py --unblock --stuck-state transcribed_not_diarized --limit 100  # Fix only transcribed but not diarized
        """
    )
    
    # Mode selection - either single content, batch from queue, or unblock
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--content',
                           help='Content ID to audit (single mode)')
    mode_group.add_argument('--content-file',
                           help='File with content IDs to audit (one per line)')
    mode_group.add_argument('--task-type',
                           help='Task type to query from queue (batch mode)')
    mode_group.add_argument('--unblock', action='store_true',
                           help='Find and unblock stuck content (unblock mode)')
    mode_group.add_argument('--audit-quality', action='store_true',
                           help='Audit all transcribed content for quality issues (requires --project)')
    
    # Batch mode options
    parser.add_argument('--status', default='failed',
                       choices=['pending', 'processing', 'completed', 'failed'],
                       help='Task status to filter by (default: failed)')
    parser.add_argument('--limit', type=int, default=10,
                       help='Maximum number of content items to process (default: 10)')
    
    # Unblock mode options
    parser.add_argument('--project',
                       help='Project to filter by in unblock mode')
    parser.add_argument('--stuck-state',
                       help='Filter by specific stuck state (e.g. transcribed_not_diarized, both_not_stitched, diarized_not_transcribed)')
    parser.add_argument('--language',
                       help='Filter by main_language (e.g. fr, en, es) - used with --audit-quality')
    
    # Common options
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed information')
    parser.add_argument('--skip-quality-check', action='store_true',
                       help='Skip transcript quality evaluation (faster)')
    parser.add_argument('--with-quality-check', action='store_true',
                       help='Force enable transcript quality evaluation in unblock mode (slower but more thorough)')
    parser.add_argument('--fix-issues', action='store_true',
                       help='Fix detected critical issues (e.g., language mismatch triggers re-transcription)')
    parser.add_argument('--config', default='config/config.yaml',
                       help='Path to config file (default: config/config.yaml)')
    
    args = parser.parse_args()
    
    try:
        # Initialize auditor
        auditor = ContentAuditor(args.config)
        
        if args.content:
            # Single content mode
            results = await auditor.audit_content(
                args.content,
                dry_run=args.dry_run,
                verbose=args.verbose,
                skip_quality_check=args.skip_quality_check,
                fix_issues=args.fix_issues
            )
            
            # Print report
            print_audit_report(results, args.verbose)
            
            # Exit code
            if results['status'] == 'error':
                return 1
            elif results['actions_taken'] and not args.dry_run:
                return 0  # Changes made
            elif results['actions_taken'] and args.dry_run:
                return 2  # Would make changes (dry run)
            else:
                return 0  # No changes needed

        elif args.content_file:
            # Batch mode from file - process list of content IDs
            content_ids = []
            with open(args.content_file, 'r') as f:
                for line in f:
                    # Strip whitespace and line numbers (e.g., "  1→abc123" -> "abc123")
                    line = line.strip()
                    if '→' in line:
                        line = line.split('→', 1)[1].strip()
                    if line:
                        content_ids.append(line)

            if not content_ids:
                print("❌ No content IDs found in file")
                return 1

            print(f"📋 Processing {len(content_ids)} content IDs from {args.content_file}")

            total_fixed = 0
            total_errors = 0
            total_skipped = 0

            with tqdm(total=len(content_ids), desc="Auditing content", unit="item") as pbar:
                for content_id in content_ids:
                    pbar.set_description(f"Auditing {content_id[:16]}...")
                    try:
                        results = await auditor.audit_content(
                            content_id,
                            dry_run=args.dry_run,
                            verbose=args.verbose,
                            skip_quality_check=False,  # Always check quality
                            fix_issues=args.fix_issues,
                            skip_state_reconciliation=True  # Only fix language issues
                        )

                        if results.get('issues_fixed'):
                            for fix in results['issues_fixed']:
                                if fix.get('fix_result', {}).get('success'):
                                    total_fixed += 1
                        elif results['status'] == 'error':
                            total_errors += 1
                        else:
                            # Check if language mismatch was detected but not fixed
                            quality = results.get('transcript_quality', {})
                            lang_issues = [i for i in quality.get('issues', []) if i.get('type') == 'language_mismatch']
                            if not lang_issues:
                                total_skipped += 1  # No language issue found

                    except Exception as e:
                        logger.error(f"Error auditing {content_id}: {e}")
                        total_errors += 1

                    pbar.update(1)
                    pbar.set_postfix({'fixed': total_fixed, 'errors': total_errors, 'skipped': total_skipped})

            print(f"\n{'='*60}")
            print(f"BATCH AUDIT COMPLETE")
            print(f"{'='*60}")
            print(f"Total processed: {len(content_ids)}")
            print(f"Fixed (language mismatch): {total_fixed}")
            print(f"Skipped (no language issue): {total_skipped}")
            print(f"Errors: {total_errors}")

            return 0 if total_errors == 0 else 1

        elif args.unblock:
            # Unblock mode - find and fix stuck content
            # Determine skip_quality_check: if --with-quality-check is set, skip_quality_check=False
            # Otherwise use --skip-quality-check value, defaulting to True for unblock mode
            skip_quality = not args.with_quality_check if args.with_quality_check else (args.skip_quality_check or True)

            unblock_results = await auditor.unblock_stuck_content(
                project=args.project,
                limit=args.limit,  # Use the specified limit
                dry_run=args.dry_run,
                verbose=args.verbose,
                stuck_state_filter=args.stuck_state,
                skip_quality_check=skip_quality,
                fix_issues=args.fix_issues
            )
            
            # Print unblock report
            print_unblock_report(unblock_results, args.verbose)
            
            # Exit codes for unblock mode
            if unblock_results['errors'] and not unblock_results['total_unblocked']:
                return 1  # Errors and no success
            elif unblock_results['total_unblocked'] > 0 and not args.dry_run:
                return 0  # Successfully unblocked content
            elif unblock_results['total_unblocked'] > 0 and args.dry_run:
                return 2  # Would unblock content (dry run)
            else:
                return 0  # No stuck content found

        elif args.audit_quality:
            # Quality audit mode - audit all transcribed content for a project
            if not args.project:
                print("❌ Error: --audit-quality requires --project")
                return 1

            quality_results = await auditor.audit_quality_by_project(
                project=args.project,
                limit=args.limit,
                dry_run=args.dry_run,
                verbose=args.verbose,
                fix_issues=args.fix_issues,
                language=args.language
            )

            # Print quality audit report
            print_quality_audit_report(quality_results, args.verbose)

            # Exit codes
            if quality_results['errors'] and not quality_results['total_fixed']:
                return 1  # Errors and no fixes
            elif quality_results['total_fixed'] > 0 and not args.dry_run:
                return 0  # Successfully fixed content
            elif quality_results['total_with_issues'] > 0 and args.dry_run:
                return 2  # Would fix content (dry run)
            else:
                return 0  # No issues found

        else:
            # Batch mode from task queue
            batch_results = await auditor.audit_multiple_from_queue(
                task_type=args.task_type,
                task_status=args.status,
                limit=args.limit,
                dry_run=args.dry_run,
                verbose=args.verbose,
                skip_quality_check=args.skip_quality_check
            )
            
            # Print batch report
            print_batch_audit_report(batch_results, args.verbose)
            
            # Exit codes for batch mode
            if batch_results['errors']:
                return 1  # Batch errors
            elif batch_results['total_actions'] > 0 and not args.dry_run:
                return 0  # Changes made
            elif batch_results['total_actions'] > 0 and args.dry_run:
                return 2  # Would make changes (dry run)
            else:
                return 0  # No changes needed
            
    except KeyboardInterrupt:
        print("\n🛑 Audit cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 