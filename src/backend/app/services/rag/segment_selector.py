"""
Segment selection component for RAG system.

Unified weighted selection approach with configurable biases:
- Recency (temporal freshness)
- Channel popularity (not yet implemented)
- User preferences (not yet implemented)
- Diversity (dissimilarity to selected)
- Centrality (representativeness)

Predefined strategy templates available for common use cases.
"""

from typing import List, Optional, Dict, Callable, Union, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime
from collections import defaultdict

from ...models.db_models import EmbeddingSegment as Segment
from .theme_extractor import Theme

from ...utils.backend_logger import get_logger
logger = get_logger("segment_selector")


@dataclass
class SamplingWeights:
    """Configurable weights for selection biases."""
    recency: float = 0.0
    channel_popularity: float = 0.0  # Not yet implemented
    user_preference: float = 0.0  # Not yet implemented
    diversity: float = 0.5
    centrality: float = 0.5

    def __post_init__(self):
        """Normalize weights to sum to 1.0."""
        total = self.recency + self.channel_popularity + self.user_preference + self.diversity + self.centrality
        if total > 0:
            self.recency /= total
            self.channel_popularity /= total
            self.user_preference /= total
            self.diversity /= total
            self.centrality /= total
        else:
            # Default to equal diversity and centrality
            self.diversity = 0.5
            self.centrality = 0.5

    @classmethod
    def diversity_strategy(cls) -> "SamplingWeights":
        """Pure diversity selection (MMR-style)."""
        return cls(diversity=1.0, centrality=0.0, recency=0.0)

    @classmethod
    def centrality_strategy(cls) -> "SamplingWeights":
        """Pure centrality selection (most representative)."""
        return cls(diversity=0.0, centrality=1.0, recency=0.0)

    @classmethod
    def recency_strategy(cls) -> "SamplingWeights":
        """Recency-focused selection with some diversity."""
        return cls(recency=0.7, diversity=0.3, centrality=0.0)

    @classmethod
    def balanced_strategy(cls) -> "SamplingWeights":
        """Balanced mix of all factors."""
        return cls(recency=0.2, diversity=0.4, centrality=0.4)

    @classmethod
    def quality_strategy(cls) -> "SamplingWeights":
        """Focus on representative, diverse content."""
        return cls(diversity=0.3, centrality=0.7, recency=0.0)


class SegmentSelector:
    """Select representative segments using weighted scoring."""

    def __init__(
        self,
        channel_popularity_fn: Optional[Callable[[str], float]] = None,
        user_preference_fn: Optional[Callable[[Segment], float]] = None
    ):
        """
        Initialize selector with optional scoring functions.

        Args:
            channel_popularity_fn: Function that takes channel_name and returns popularity score (0-1)
            user_preference_fn: Function that takes Segment and returns user preference score (0-1)
        """
        self.channel_popularity_fn = channel_popularity_fn
        self.user_preference_fn = user_preference_fn

    def select(
        self,
        segments: List[Segment],
        n: int = 10,
        weights: Optional[SamplingWeights] = None,
        strategy: Optional[str] = None,
        rank_only: bool = False
    ) -> Union[List[Segment], List[Tuple[Segment, float]]]:
        """
        Select or rank segments using weighted scoring.

        Args:
            segments: Segments to select from
            n: Number of segments to select (or rank if rank_only=True)
            weights: Custom selection weights (overrides strategy)
            strategy: Predefined strategy name (diversity, centrality, recency, balanced, quality)
            rank_only: If True, return segments with scores instead of just top N

        Returns:
            If rank_only=False: List of selected segments (top N)
            If rank_only=True: List of tuples (segment, score) sorted by score descending
        """
        if len(segments) <= n:
            return segments

        # Determine weights
        if weights is None:
            if strategy == "diversity":
                weights = SamplingWeights.diversity_strategy()
            elif strategy == "centrality":
                weights = SamplingWeights.centrality_strategy()
            elif strategy == "recency":
                weights = SamplingWeights.recency_strategy()
            elif strategy == "balanced":
                weights = SamplingWeights.balanced_strategy()
            elif strategy == "quality":
                weights = SamplingWeights.quality_strategy()
            else:
                weights = SamplingWeights.balanced_strategy()
                logger.info(f"No strategy specified, using balanced")

        logger.info(f"Selecting with weights: recency={weights.recency:.2f}, "
                   f"diversity={weights.diversity:.2f}, centrality={weights.centrality:.2f}")

        # Get embeddings
        valid_segments = [seg for seg in segments if seg.embedding is not None]

        if len(valid_segments) == 0:
            logger.warning("No segments with embeddings available")
            return []

        if len(valid_segments) <= n:
            return valid_segments

        embeddings = np.vstack([seg.embedding for seg in valid_segments])

        # Compute base scores
        recency_scores = self._compute_recency_scores(valid_segments) if weights.recency > 0 else None
        popularity_scores = self._compute_popularity_scores(valid_segments) if weights.channel_popularity > 0 else None
        preference_scores = self._compute_preference_scores(valid_segments) if weights.user_preference > 0 else None
        centrality_scores = self._compute_centrality_scores(embeddings) if weights.centrality > 0 else None

        # If rank_only, compute scores for all segments without diversity (which requires selection order)
        if rank_only:
            # Compute static scores for all segments
            all_scores = []
            for idx in range(len(valid_segments)):
                static_score = self._compute_static_scores(
                    idx, valid_segments, weights,
                    recency_scores, popularity_scores, preference_scores, centrality_scores
                )
                all_scores.append((valid_segments[idx], static_score))

            # Sort by score descending and return top N
            all_scores.sort(key=lambda x: x[1], reverse=True)
            return all_scores[:n]

        # Greedy selection with weighted scoring (includes diversity)
        selected_indices = []
        remaining_indices = set(range(len(valid_segments)))

        # Select first segment based on static scores
        first_scores = np.array([
            self._compute_static_scores(
                idx, valid_segments, weights,
                recency_scores, popularity_scores, preference_scores, centrality_scores
            )
            for idx in range(len(valid_segments))
        ])
        first_idx = int(np.argmax(first_scores))
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Select remaining segments
        while len(selected_indices) < n and remaining_indices:
            best_score = -float('inf')
            best_idx = None

            for idx in remaining_indices:
                # Static scores (recency, popularity, preference, centrality)
                static_score = self._compute_static_scores(
                    idx, valid_segments, weights,
                    recency_scores, popularity_scores, preference_scores, centrality_scores
                )

                # Diversity score (depends on already selected)
                diversity_score = 0.0
                if weights.diversity > 0:
                    selected_embeddings = embeddings[selected_indices]
                    similarities = np.dot(selected_embeddings, embeddings[idx])
                    max_similarity = np.max(similarities)
                    diversity_score = 1 - max_similarity

                # Combined score
                total_score = static_score + weights.diversity * diversity_score

                if total_score > best_score:
                    best_score = total_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else:
                break

        return [valid_segments[i] for i in selected_indices]

    def select_from_theme(
        self,
        theme: Theme,
        n: int = 10,
        weights: Optional[SamplingWeights] = None,
        strategy: Optional[str] = None,
        rank_only: bool = False
    ) -> Union[List[Segment], List[Tuple[Segment, float]]]:
        """
        Select or rank segments from a theme.

        Args:
            theme: Theme to select from
            n: Number of segments to select (or rank if rank_only=True)
            weights: Custom selection weights
            strategy: Predefined strategy name
            rank_only: If True, return segments with scores

        Returns:
            If rank_only=False: List of selected segments
            If rank_only=True: List of tuples (segment, score)
        """
        return self.select(theme.segments, n=n, weights=weights, strategy=strategy, rank_only=rank_only)

    def select_balanced_by_group(
        self,
        segments: List[Segment],
        balance_by: str = "channel",
        n_per_group: int = 5,
        weights: Optional[SamplingWeights] = None,
        strategy: Optional[str] = None
    ) -> List[Segment]:
        """
        Select balanced across groups using weighted selection within each group.

        Args:
            segments: Segments to select from
            balance_by: Field to balance by (channel, speaker, date)
            n_per_group: Number of segments per group
            weights: Custom selection weights for within-group selection
            strategy: Predefined strategy for within-group selection

        Returns:
            List of balanced selected segments
        """
        if not segments:
            return []

        # Group segments
        groups = defaultdict(list)

        for seg in segments:
            if balance_by == "channel":
                key = getattr(seg, 'channel_name', 'unknown')
            elif balance_by == "speaker":
                key = getattr(seg, 'speaker_name', 'unknown')
            elif balance_by == "date":
                if hasattr(seg, 'start_time') and seg.start_time:
                    if isinstance(seg.start_time, datetime):
                        key = seg.start_time.date().isoformat()
                    else:
                        key = 'unknown'
                else:
                    key = 'unknown'
            else:
                raise ValueError(f"Unknown balance_by field: {balance_by}")

            groups[key].append(seg)

        # Select from each group using weighted selection
        selected = []
        for group_key, group_segments in groups.items():
            if len(group_segments) <= n_per_group:
                selected.extend(group_segments)
            else:
                group_selection = self.select(
                    group_segments,
                    n=n_per_group,
                    weights=weights,
                    strategy=strategy
                )
                selected.extend(group_selection)

        logger.info(f"Selected {len(selected)} segments from {len(groups)} groups "
                   f"(balance_by={balance_by}, n_per_group={n_per_group})")

        return selected

    def _compute_recency_scores(self, segments: List[Segment]) -> np.ndarray:
        """Compute normalized recency scores (0-1, newer = higher)."""
        timestamps = []
        for seg in segments:
            if hasattr(seg, 'start_time') and seg.start_time is not None:
                if isinstance(seg.start_time, datetime):
                    timestamps.append(seg.start_time.timestamp())
                else:
                    timestamps.append(float(seg.start_time))
            else:
                timestamps.append(0.0)

        timestamps = np.array(timestamps)
        min_ts = timestamps.min()
        max_ts = timestamps.max()

        if max_ts > min_ts:
            scores = (timestamps - min_ts) / (max_ts - min_ts)
        else:
            scores = np.ones(len(timestamps)) * 0.5

        return scores

    def _compute_popularity_scores(self, segments: List[Segment]) -> Optional[np.ndarray]:
        """Compute channel popularity scores (0-1)."""
        if not self.channel_popularity_fn:
            return None

        scores = []
        for seg in segments:
            channel_name = getattr(seg, 'channel_name', None)
            if channel_name:
                scores.append(self.channel_popularity_fn(channel_name))
            else:
                scores.append(0.0)

        return np.array(scores)

    def _compute_preference_scores(self, segments: List[Segment]) -> Optional[np.ndarray]:
        """Compute user preference scores (0-1)."""
        if not self.user_preference_fn:
            return None

        scores = [self.user_preference_fn(seg) for seg in segments]
        return np.array(scores)

    def _compute_centrality_scores(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute centrality scores (similarity to centroid, 0-1)."""
        centroid = np.mean(embeddings, axis=0)
        similarities = np.dot(embeddings, centroid)

        # Normalize to 0-1
        min_sim = similarities.min()
        max_sim = similarities.max()

        if max_sim > min_sim:
            scores = (similarities - min_sim) / (max_sim - min_sim)
        else:
            scores = np.ones(len(similarities)) * 0.5

        return scores

    def _compute_static_scores(
        self,
        idx: int,
        segments: List[Segment],
        weights: SamplingWeights,
        recency_scores: Optional[np.ndarray],
        popularity_scores: Optional[np.ndarray],
        preference_scores: Optional[np.ndarray],
        centrality_scores: Optional[np.ndarray]
    ) -> float:
        """Compute weighted static score for a segment."""
        score = 0.0

        if weights.recency > 0 and recency_scores is not None:
            score += weights.recency * recency_scores[idx]

        if weights.channel_popularity > 0 and popularity_scores is not None:
            score += weights.channel_popularity * popularity_scores[idx]

        if weights.user_preference > 0 and preference_scores is not None:
            score += weights.user_preference * preference_scores[idx]

        if weights.centrality > 0 and centrality_scores is not None:
            score += weights.centrality * centrality_scores[idx]

        return score


# Convenience functions for common strategies
def select_diverse(segments: List[Segment], n: int = 10) -> List[Segment]:
    """Select with maximum diversity."""
    selector = SegmentSelector()
    return selector.select(segments, n=n, strategy="diversity")


def select_representative(segments: List[Segment], n: int = 10) -> List[Segment]:
    """Select most representative/central segments."""
    selector = SegmentSelector()
    return selector.select(segments, n=n, strategy="centrality")


def select_recent(segments: List[Segment], n: int = 10) -> List[Segment]:
    """Select with recency bias."""
    selector = SegmentSelector()
    return selector.select(segments, n=n, strategy="recency")


def select_balanced(segments: List[Segment], n: int = 10) -> List[Segment]:
    """Select with balanced weights."""
    selector = SegmentSelector()
    return selector.select(segments, n=n, strategy="balanced")
