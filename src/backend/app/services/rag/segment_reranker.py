"""
Segment Reranker
================

Reranks retrieved segments based on multiple factors:
- Channel popularity (importance_score from channels table)
- Recency (publish_date freshness)
- Single speaker ratio (60%+ of segment from one speaker)
- Named speaker (speaker has identified name via speaker_identities)

Also enforces diversity constraints:
- Best match per episode (one segment per content_id)
- Channel diversity (limit segments per channel)

This component sits AFTER semantic retrieval and BEFORE final selection.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np

from ...utils.backend_logger import get_logger
logger = get_logger("segment_reranker")


@dataclass
class RerankerWeights:
    """Configurable weights for reranking factors."""
    similarity: float = 0.4       # Base semantic similarity
    popularity: float = 0.2       # Channel importance score
    recency: float = 0.2          # Publish date freshness
    single_speaker: float = 0.1   # Segment has dominant speaker (60%+)
    named_speaker: float = 0.1    # Speaker has identified name
    similarity_floor: float = 0.0 # Minimum similarity to include (0 = no floor)

    def __post_init__(self):
        """Normalize weights to sum to 1.0 (excluding similarity_floor)."""
        total = (self.similarity + self.popularity + self.recency +
                 self.single_speaker + self.named_speaker)
        if total > 0:
            self.similarity /= total
            self.popularity /= total
            self.recency /= total
            self.single_speaker /= total
            self.named_speaker /= total

    @classmethod
    def topic_summary(cls) -> "RerankerWeights":
        """
        Preset for topic summaries: heavily weight channel importance and recency.

        Once content is semantically relevant (above similarity_floor), prioritize:
        - Recent content from popular/important channels
        - Similarity acts as a gate (floor=0.45), not the primary ranking signal

        Weights: similarity=0.15, popularity=0.35, recency=0.35, speaker signals=0.15
        Floor: 0.45 (drops segments with similarity < 45%)
        """
        return cls(
            similarity=0.15,      # Reduced: similarity is a gate, not primary signal
            popularity=0.35,      # Boosted: channel importance dominates
            recency=0.35,         # Boosted: recent content prioritized
            single_speaker=0.08,  # Slight reduction
            named_speaker=0.07,   # Slight reduction
            similarity_floor=0.45 # Gate: must be at least 45% similar
        )

    @classmethod
    def balanced(cls) -> "RerankerWeights":
        """Default balanced weights (no similarity floor)."""
        return cls()  # Uses default values


@dataclass
class DiversityConstraints:
    """Configurable diversity constraints."""
    best_per_episode: bool = True           # Take best segment per episode (content_id)
    max_per_channel: Optional[int] = None   # Limit segments per channel (None = no limit)
    min_channels: Optional[int] = None      # Minimum channels to include (None = no minimum)


class SegmentReranker:
    """
    Reranks segments based on multiple quality signals.

    Design:
    - Receives raw search results (dicts with segment_id, similarity, etc.)
    - Enriches with channel importance, speaker info (lazy loaded)
    - Computes composite rerank scores
    - Applies diversity constraints
    - Returns reranked list

    Usage:
        reranker = SegmentReranker(db_session)
        reranked = reranker.rerank(
            segments,
            weights=RerankerWeights(popularity=0.3),
            diversity=DiversityConstraints(best_per_episode=True)
        )
    """

    def __init__(self, db_session=None, skip_enrichment: bool = False):
        """
        Initialize reranker.

        Args:
            db_session: SQLAlchemy session for enrichment queries
            skip_enrichment: If True, skip database lookups for channel/speaker enrichment
                            (faster but popularity and named_speaker scores will be 0)
        """
        self.db_session = db_session
        self.skip_enrichment = skip_enrichment
        self._channel_scores_cache: Dict[str, float] = {}
        self._speaker_names_cache: Dict[str, Optional[str]] = {}
        self._max_importance_score: Optional[float] = None
        self._db_connection = None  # Lazy psycopg2 connection

    def rerank(
        self,
        segments: List[Dict[str, Any]],
        weights: Optional[RerankerWeights] = None,
        diversity: Optional[DiversityConstraints] = None,
        time_window_days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Rerank segments using composite scoring and diversity constraints.

        Args:
            segments: List of segment dicts from search (must have similarity, content_id, etc.)
            weights: Scoring weights (default: balanced)
            diversity: Diversity constraints (default: best per episode)
            time_window_days: Time window for recency scoring normalization

        Returns:
            Reranked list of segment dicts with added 'rerank_score' field
        """
        if not segments:
            return []

        weights = weights or RerankerWeights()
        diversity = diversity or DiversityConstraints()

        logger.info(f"Reranking {len(segments)} segments with weights: "
                   f"sim={weights.similarity:.2f}, pop={weights.popularity:.2f}, "
                   f"rec={weights.recency:.2f}, single={weights.single_speaker:.2f}, "
                   f"named={weights.named_speaker:.2f}, floor={weights.similarity_floor:.2f}")

        # Apply similarity floor if configured
        if weights.similarity_floor > 0:
            original_count = len(segments)
            segments = [s for s in segments if s.get('similarity', 0) >= weights.similarity_floor]
            filtered_count = original_count - len(segments)
            if filtered_count > 0:
                logger.info(f"Similarity floor {weights.similarity_floor:.2f}: "
                           f"filtered {filtered_count} segments, {len(segments)} remaining")

        if not segments:
            logger.warning("No segments remaining after similarity floor filtering")
            return []

        # Enrich segments with scoring data
        enriched = self._enrich_segments(segments)

        # Compute recency normalization bounds
        now = datetime.now(timezone.utc)
        oldest_date = now - timedelta(days=time_window_days)

        # Score each segment
        scored = []
        for seg in enriched:
            score = self._compute_score(seg, weights, now, oldest_date)
            seg['rerank_score'] = score
            scored.append(seg)

        # Sort by rerank score
        scored.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Apply diversity constraints
        if diversity.best_per_episode:
            scored = self._best_per_episode(scored)

        if diversity.max_per_channel:
            scored = self._limit_per_channel(scored, diversity.max_per_channel)

        logger.info(f"Reranking complete: {len(scored)} segments after diversity constraints")
        return scored

    def _get_db_connection(self):
        """Get or create a psycopg2 connection for fast queries."""
        if self._db_connection is None:
            import psycopg2
            from src.backend.app.config.database import get_db_config
            self._db_connection = psycopg2.connect(**get_db_config())
        return self._db_connection

    def _enrich_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich segments with channel popularity and speaker info.

        Lazy loads data from database as needed.
        """
        # Collect unique channel URLs and speaker hashes for batch lookup
        channel_urls = set()
        speaker_hashes = set()

        for seg in segments:
            if seg.get('channel_url'):
                channel_urls.add(seg['channel_url'])
            if seg.get('speaker_hashes'):
                speaker_hashes.update(seg['speaker_hashes'])

        # Batch load channel importance scores (skip if enrichment disabled)
        if channel_urls and not self.skip_enrichment:
            self._load_channel_scores(channel_urls)

        # Batch load speaker names (skip if enrichment disabled)
        if speaker_hashes and not self.skip_enrichment:
            self._load_speaker_names(speaker_hashes)

        # Enrich each segment
        for seg in segments:
            # Channel popularity (0-1 normalized)
            channel_url = seg.get('channel_url')
            if channel_url and channel_url in self._channel_scores_cache:
                raw_score = self._channel_scores_cache[channel_url]
                # Normalize to 0-1 using log scale
                if self._max_importance_score and self._max_importance_score > 0:
                    # Use log scale since importance varies widely (0 to 90000+)
                    seg['_channel_popularity'] = min(1.0, np.log1p(raw_score) / np.log1p(self._max_importance_score))
                else:
                    seg['_channel_popularity'] = 0.0
            else:
                seg['_channel_popularity'] = 0.0

            # Single speaker analysis
            speaker_hashes = seg.get('speaker_hashes', [])
            if speaker_hashes:
                # Single speaker if only one hash, or could analyze speaker_positions
                # For now: single if array length is 1
                seg['_is_single_speaker'] = len(speaker_hashes) == 1

                # Check if any speaker is named
                has_named = any(
                    self._speaker_names_cache.get(h)
                    for h in speaker_hashes
                )
                seg['_has_named_speaker'] = has_named
            else:
                seg['_is_single_speaker'] = False
                seg['_has_named_speaker'] = False

        return segments

    def _load_channel_scores(self, channel_urls: set):
        """Batch load channel importance scores using psycopg2 for speed."""
        # Filter to URLs we haven't cached yet
        uncached = [url for url in channel_urls if url not in self._channel_scores_cache]
        if not uncached:
            return

        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT primary_url, importance_score
                FROM channels
                WHERE primary_url = ANY(%s)
            """, (uncached,))

            for row in cursor:
                url, score = row
                self._channel_scores_cache[url] = score or 0.0

            cursor.close()

            # Track max for normalization
            if self._channel_scores_cache:
                self._max_importance_score = max(self._channel_scores_cache.values())

            # Cache misses as 0
            for url in uncached:
                if url not in self._channel_scores_cache:
                    self._channel_scores_cache[url] = 0.0

            logger.debug(f"Loaded {len(uncached)} channel scores, max={self._max_importance_score}")

        except Exception as e:
            logger.warning(f"Failed to load channel scores: {e}")
            for url in uncached:
                self._channel_scores_cache[url] = 0.0

    def _load_speaker_names(self, speaker_hashes: set):
        """Batch load speaker identity names using psycopg2 for speed."""
        # Filter to hashes we haven't cached yet
        uncached = [h for h in speaker_hashes if h not in self._speaker_names_cache]
        if not uncached:
            return

        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            # Join speakers to speaker_identities to get names
            cursor.execute("""
                SELECT DISTINCT s.speaker_hash, si.primary_name
                FROM speakers s
                JOIN speaker_identities si ON s.speaker_identity_id = si.id
                WHERE s.speaker_hash = ANY(%s)
                  AND si.primary_name IS NOT NULL
            """, (uncached,))

            named_count = 0
            for row in cursor:
                hash_, name = row
                self._speaker_names_cache[hash_] = name
                named_count += 1

            cursor.close()

            # Cache misses as None
            for h in uncached:
                if h not in self._speaker_names_cache:
                    self._speaker_names_cache[h] = None

            logger.debug(f"Loaded speaker names: {named_count} named out of {len(uncached)}")

        except Exception as e:
            logger.warning(f"Failed to load speaker names: {e}")
            for h in uncached:
                self._speaker_names_cache[h] = None

    def _compute_score(
        self,
        segment: Dict[str, Any],
        weights: RerankerWeights,
        now: datetime,
        oldest_date: datetime
    ) -> float:
        """Compute composite rerank score for a segment."""
        score = 0.0

        # Similarity score (already 0-1 from search)
        similarity = segment.get('similarity', 0.0)
        score += weights.similarity * similarity

        # Channel popularity (0-1 normalized)
        popularity = segment.get('_channel_popularity', 0.0)
        score += weights.popularity * popularity

        # Recency score (0-1, newer = higher)
        publish_date = segment.get('publish_date')
        if publish_date:
            if isinstance(publish_date, str):
                try:
                    publish_date = datetime.fromisoformat(publish_date.replace('Z', '+00:00'))
                except:
                    publish_date = None

            if publish_date:
                # Ensure timezone aware
                if publish_date.tzinfo is None:
                    publish_date = publish_date.replace(tzinfo=timezone.utc)

                # Linear decay from 1.0 (now) to 0.0 (oldest_date)
                time_range = (now - oldest_date).total_seconds()
                if time_range > 0:
                    age = (now - publish_date).total_seconds()
                    recency = max(0.0, 1.0 - (age / time_range))
                else:
                    recency = 1.0
                score += weights.recency * recency

        # Single speaker bonus
        if segment.get('_is_single_speaker'):
            score += weights.single_speaker * 1.0

        # Named speaker bonus
        if segment.get('_has_named_speaker'):
            score += weights.named_speaker * 1.0

        return score

    def _best_per_episode(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Keep only the best segment per episode (content_id).

        Assumes segments are already sorted by rerank_score descending.
        """
        seen_content_ids = set()
        result = []

        for seg in segments:
            content_id = seg.get('content_id')
            if content_id not in seen_content_ids:
                seen_content_ids.add(content_id)
                result.append(seg)

        logger.debug(f"Best per episode: {len(segments)} -> {len(result)} segments")
        return result

    def _limit_per_channel(
        self,
        segments: List[Dict[str, Any]],
        max_per_channel: int
    ) -> List[Dict[str, Any]]:
        """
        Limit segments per channel while maintaining score order.
        """
        channel_counts = defaultdict(int)
        result = []

        for seg in segments:
            channel = seg.get('channel_name', 'unknown')
            if channel_counts[channel] < max_per_channel:
                channel_counts[channel] += 1
                result.append(seg)

        logger.debug(f"Limit per channel ({max_per_channel}): {len(segments)} -> {len(result)} segments")
        return result


# Convenience function for simple reranking
def rerank_segments(
    segments: List[Dict[str, Any]],
    db_session=None,
    best_per_episode: bool = True,
    time_window_days: int = 30
) -> List[Dict[str, Any]]:
    """
    Simple reranking with default weights and best-per-episode diversity.

    Args:
        segments: Segment dicts from search
        db_session: SQLAlchemy session for enrichment
        best_per_episode: Whether to keep only best segment per episode
        time_window_days: Time window for recency normalization

    Returns:
        Reranked segments
    """
    reranker = SegmentReranker(db_session)
    return reranker.rerank(
        segments,
        weights=RerankerWeights(),
        diversity=DiversityConstraints(best_per_episode=best_per_episode),
        time_window_days=time_window_days
    )
