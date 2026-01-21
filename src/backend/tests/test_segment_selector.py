"""
Tests for SegmentSelector.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from datetime import datetime, timedelta

from app.services.rag.segment_selector import (
    SegmentSelector,
    SamplingWeights,
    select_diverse,
    select_representative,
    select_recent,
    select_balanced
)
from app.services.rag.theme_extractor import Theme
from src.database.models import EmbeddingSegment as Segment


@pytest.fixture
def mock_segments():
    """Create mock segments with embeddings and metadata."""
    segments = []
    base_time = datetime.now()

    # Create 30 segments with varying properties
    for i in range(30):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.text = f"Segment text {i}"
        seg.channel_name = f"Channel_{i % 5}"  # 5 different channels
        seg.speaker_name = f"Speaker_{i % 3}"  # 3 different speakers
        seg.start_time = base_time - timedelta(days=i)  # Vary recency
        seg.duration = 30.0 + i  # Varying duration
        seg.embedding = np.random.randn(1024) + np.array([i / 30.0] * 1024)  # Gradual shift
        segments.append(seg)

    return segments


@pytest.fixture
def mock_theme(mock_segments):
    """Create mock theme."""
    return Theme(
        theme_id="test_theme",
        theme_name="Test Theme",
        segments=mock_segments,
        representative_segments=mock_segments[:5]
    )


def test_sampling_weights_normalization():
    """Test that weights are normalized to sum to 1.0."""
    weights = SamplingWeights(recency=2.0, diversity=3.0, centrality=5.0)

    total = weights.recency + weights.diversity + weights.centrality
    assert abs(total - 1.0) < 0.01


def test_sampling_weights_default():
    """Test default weights."""
    weights = SamplingWeights()

    assert weights.diversity == 0.5
    assert weights.centrality == 0.5
    assert weights.recency == 0.0


def test_sampling_weights_strategies():
    """Test predefined strategy templates."""
    # Diversity strategy
    div_weights = SamplingWeights.diversity_strategy()
    assert div_weights.diversity == 1.0
    assert div_weights.centrality == 0.0

    # Centrality strategy
    cent_weights = SamplingWeights.centrality_strategy()
    assert cent_weights.centrality == 1.0
    assert cent_weights.diversity == 0.0

    # Recency strategy
    rec_weights = SamplingWeights.recency_strategy()
    assert rec_weights.recency > 0.5

    # Balanced strategy
    bal_weights = SamplingWeights.balanced_strategy()
    assert bal_weights.recency > 0
    assert bal_weights.diversity > 0
    assert bal_weights.centrality > 0


def test_sample_basic(mock_segments):
    """Test selection."""
    selector = SegmentSelector()

    selected = selector.select(mock_segments, n=10, strategy="balanced")

    assert len(selected) == 10
    assert all(seg in mock_segments for seg in selected)
    assert len(set(seg.id for seg in selected)) == 10  # No duplicates


def test_sample_fewer_than_n(mock_segments):
    """Test selection when fewer segments than n."""
    selector = SegmentSelector()

    # Request more than available
    selected = selector.select(mock_segments[:5], n=10, strategy="balanced")

    assert len(selected) == 5


def test_sample_diversity_strategy(mock_segments):
    """Test selection."""
    selector = SegmentSelector()

    selected = selector.select(mock_segments, n=10, strategy="diversity")

    assert len(selected) == 10

    # Check that selected segments are relatively diverse
    embeddings = np.vstack([seg.embedding for seg in selected])

    # Normalize embeddings for proper cosine similarity
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    similarities = np.dot(embeddings_norm, embeddings_norm.T)

    # Off-diagonal similarities should be lower for diverse selection
    off_diag = similarities[np.triu_indices_from(similarities, k=1)]
    assert np.mean(off_diag) < 0.9  # Not too similar


def test_sample_centrality_strategy(mock_segments):
    """Test selection."""
    selector = SegmentSelector()

    selected = selector.select(mock_segments, n=10, strategy="centrality")

    assert len(selected) == 10

    # Centrality selections should be close to centroid
    all_embeddings = np.vstack([seg.embedding for seg in mock_segments])
    centroid = np.mean(all_embeddings, axis=0)

    selected_embeddings = np.vstack([seg.embedding for seg in selected])
    selected_similarities = np.dot(selected_embeddings, centroid)

    # Should have high average similarity to centroid
    assert np.mean(selected_similarities) > 0.5


def test_sample_recency_strategy(mock_segments):
    """Test selection."""
    selector = SegmentSelector()

    selected = selector.select(mock_segments, n=10, strategy="recency")

    assert len(selected) == 10

    # Should favor recent segments (lower index = more recent)
    avg_index = np.mean([seg.id for seg in selected])
    assert avg_index < 15  # Should be biased toward earlier indices


def test_sample_custom_weights(mock_segments):
    """Test selection with custom weights."""
    selector = SegmentSelector()

    custom_weights = SamplingWeights(
        recency=0.5,
        diversity=0.3,
        centrality=0.2
    )

    selected = selector.select(mock_segments, n=10, weights=custom_weights)

    assert len(selected) == 10


def test_select_from_theme(mock_theme):
    """Test selection from a theme."""
    selector = SegmentSelector()

    selected = selector.select_from_theme(mock_theme, n=10, strategy="balanced")

    assert len(selected) == 10
    assert all(seg in mock_theme.segments for seg in selected)


def test_select_balanced_by_group_channel(mock_segments):
    """Test selection across channels."""
    selector = SegmentSelector()

    selected = selector.select_balanced_by_group(
        mock_segments,
        balance_by="channel",
        n_per_group=3,
        strategy="centrality"
    )

    # Should have segments from multiple channels
    channels = set(seg.channel_name for seg in selected)
    assert len(channels) >= 3

    # Should respect n_per_group limit
    from collections import Counter
    channel_counts = Counter(seg.channel_name for seg in selected)
    assert all(count <= 3 for count in channel_counts.values())


def test_select_balanced_by_group_speaker(mock_segments):
    """Test selection across speakers."""
    selector = SegmentSelector()

    selected = selector.select_balanced_by_group(
        mock_segments,
        balance_by="speaker",
        n_per_group=5,
        strategy="diversity"
    )

    # Should have segments from multiple speakers
    speakers = set(seg.speaker_name for seg in selected)
    assert len(speakers) >= 2


def test_select_balanced_by_group_date(mock_segments):
    """Test selection across dates."""
    selector = SegmentSelector()

    selected = selector.select_balanced_by_group(
        mock_segments,
        balance_by="date",
        n_per_group=2,
        strategy="balanced"
    )

    # Should have segments from multiple dates
    dates = set(seg.start_time.date() for seg in selected)
    assert len(dates) >= 3


def test_sample_with_channel_popularity():
    """Test selection with channel popularity function."""
    np.random.seed(42)  # Fixed seed for reproducibility

    # Create segments with specific channels
    segments = []
    for i in range(20):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.channel_name = "popular" if i < 10 else "unpopular"
        seg.embedding = np.random.randn(1024)
        seg.start_time = datetime.now()
        segments.append(seg)

    # Define popularity function
    def popularity_fn(channel_name):
        return 1.0 if channel_name == "popular" else 0.1

    selector = SegmentSelector(channel_popularity_fn=popularity_fn)

    # Use weights that favor popularity
    weights = SamplingWeights(
        channel_popularity=0.7,
        diversity=0.3,
        centrality=0.0
    )

    selected = selector.select(segments, n=10, weights=weights)

    # Should favor popular channel
    popular_count = sum(1 for seg in selected if seg.channel_name == "popular")
    assert popular_count >= 6  # At least 60% from popular channel


def test_sample_with_user_preference():
    """Test selection with user preference function."""
    np.random.seed(42)  # Fixed seed for reproducibility

    # Create segments
    segments = []
    for i in range(20):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.text = f"preferred_{i}" if i < 10 else f"other_{i}"
        seg.embedding = np.random.randn(1024)
        seg.start_time = datetime.now()
        segments.append(seg)

    # Define preference function
    def preference_fn(seg):
        return 1.0 if "preferred" in seg.text else 0.1

    selector = SegmentSelector(user_preference_fn=preference_fn)

    # Use weights that favor preferences
    weights = SamplingWeights(
        user_preference=0.8,
        diversity=0.2,
        centrality=0.0
    )

    selected = selector.select(segments, n=10, weights=weights)

    # Should favor preferred segments
    preferred_count = sum(1 for seg in selected if "preferred" in seg.text)
    assert preferred_count >= 6  # At least 60% from preferred


def test_compute_recency_scores(mock_segments):
    """Test recency score computation."""
    selector = SegmentSelector()

    scores = selector._compute_recency_scores(mock_segments)

    # Should return normalized scores
    assert len(scores) == len(mock_segments)
    assert 0 <= scores.min() <= 1
    assert 0 <= scores.max() <= 1

    # More recent segments (lower index) should have higher scores
    assert scores[0] > scores[-1]


def test_compute_centrality_scores(mock_segments):
    """Test centrality score computation."""
    selector = SegmentSelector()

    embeddings = np.vstack([seg.embedding for seg in mock_segments])
    scores = selector._compute_centrality_scores(embeddings)

    # Should return normalized scores
    assert len(scores) == len(mock_segments)
    assert 0 <= scores.min() <= 1
    assert 0 <= scores.max() <= 1


def test_convenience_functions(mock_segments):
    """Test convenience functions."""
    # Test select_diverse
    diverse = select_diverse(mock_segments, n=5)
    assert len(diverse) == 5

    # Test select_representative
    representative = select_representative(mock_segments, n=5)
    assert len(representative) == 5

    # Test select_recent
    recent = select_recent(mock_segments, n=5)
    assert len(recent) == 5

    # Test select_balanced
    balanced = select_balanced(mock_segments, n=5)
    assert len(balanced) == 5


def test_sample_no_embeddings():
    """Test selection with segments without embeddings."""
    segments = []
    for i in range(10):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.text = f"Text {i}"
        seg.embedding = None  # No embedding
        seg.start_time = datetime.now()
        segments.append(seg)

    selector = SegmentSelector()
    selected = selector.select(segments, n=5, strategy="balanced")

    # Should return empty or handle gracefully
    assert len(selected) == 0


def test_sample_mixed_embeddings(mock_segments):
    """Test selection with some segments missing embeddings."""
    # Remove embeddings from half the segments
    for i in range(0, len(mock_segments), 2):
        mock_segments[i].embedding = None

    selector = SegmentSelector()
    selected = selector.select(mock_segments, n=5, strategy="balanced")

    # Should only select from segments with embeddings
    assert len(selected) <= 5
    assert all(seg.embedding is not None for seg in selected)


def test_invalid_balance_by():
    """Test invalid balance_by parameter."""
    segments = [Mock(spec=Segment) for _ in range(10)]

    selector = SegmentSelector()

    with pytest.raises(ValueError, match="Unknown balance_by field"):
        selector.select_balanced_by_group(
            segments,
            balance_by="invalid_field",
            n_per_group=2
        )


def test_rank_only_mode(mock_segments):
    """Test rank_only mode returns scored segments."""
    selector = SegmentSelector()

    # Use rank_only=True
    ranked = selector.select(mock_segments, n=10, strategy="centrality", rank_only=True)

    # Should return list of tuples
    assert len(ranked) == 10
    assert all(isinstance(item, tuple) for item in ranked)
    assert all(len(item) == 2 for item in ranked)

    # First item should be (segment, score)
    seg, score = ranked[0]
    assert hasattr(seg, 'id')
    assert isinstance(score, (int, float))

    # Scores should be in descending order
    scores = [score for _, score in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rank_only_vs_select(mock_segments):
    """Test that rank_only and select modes are consistent."""
    selector = SegmentSelector()

    # Get selections (normal mode)
    selected = selector.select(mock_segments, n=5, strategy="centrality", rank_only=False)

    # Get rankings (rank_only mode)
    ranked = selector.select(mock_segments, n=5, strategy="centrality", rank_only=True)

    # Ranked should return tuples
    assert all(isinstance(item, tuple) for item in ranked)

    # Selected should return segments only
    assert all(not isinstance(item, tuple) for item in selected)

    # Both should have same length
    assert len(selected) == len(ranked) == 5
