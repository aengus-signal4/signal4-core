"""
Tests for ThemeExtractor.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import asyncio

from app.services.rag.theme_extractor import ThemeExtractor, Theme
from src.database.models import EmbeddingSegment as Segment


@pytest.fixture
def mock_segments():
    """Create mock segments with embeddings."""
    segments = []

    # Group 1: Politics (10 segments)
    for i in range(10):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.text = f"Political discussion about policy {i}"
        seg.embedding = np.random.randn(1024) + np.array([1.0] * 512 + [0.0] * 512)  # Similar embeddings
        segments.append(seg)

    # Group 2: Economy (10 segments)
    for i in range(10, 20):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.text = f"Economic analysis of markets {i}"
        seg.embedding = np.random.randn(1024) + np.array([0.0] * 512 + [1.0] * 512)  # Different cluster
        segments.append(seg)

    # Group 3: Technology (5 segments)
    for i in range(20, 25):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.text = f"Technology innovation {i}"
        seg.embedding = np.random.randn(1024) + np.array([1.0] * 256 + [0.0] * 768)  # Third cluster
        segments.append(seg)

    return segments


@pytest.fixture
def mock_llm_service():
    """Create mock LLM service."""
    llm = Mock()
    llm.batch_convert_queries_to_embeddings = Mock(return_value=[
        np.random.randn(1024) for _ in range(3)
    ])
    return llm


def test_extract_by_clustering_hdbscan(mock_segments):
    """Test HDBSCAN clustering."""
    extractor = ThemeExtractor()

    themes = extractor.extract_by_clustering(
        segments=mock_segments,
        method="hdbscan",
        min_cluster_size=5
    )

    # Should find at least 2 themes (politics and economy)
    assert len(themes) >= 2
    assert all(isinstance(theme, Theme) for theme in themes)
    assert all(len(theme.segments) >= 5 for theme in themes)
    assert all(len(theme.representative_segments) <= 10 for theme in themes)


def test_extract_by_clustering_kmeans(mock_segments):
    """Test k-means clustering."""
    extractor = ThemeExtractor()

    themes = extractor.extract_by_clustering(
        segments=mock_segments,
        method="kmeans",
        n_clusters=3,
        min_cluster_size=3
    )

    # Should find exactly 3 themes
    assert len(themes) == 3
    assert all(isinstance(theme, Theme) for theme in themes)
    assert sum(len(theme.segments) for theme in themes) <= len(mock_segments)


def test_extract_by_clustering_agglomerative(mock_segments):
    """Test agglomerative clustering."""
    extractor = ThemeExtractor()

    themes = extractor.extract_by_clustering(
        segments=mock_segments,
        method="agglomerative",
        n_clusters=2,
        min_cluster_size=5
    )

    # Should find 2 themes
    assert len(themes) == 2
    assert all(isinstance(theme, Theme) for theme in themes)


def test_extract_by_clustering_too_few_segments():
    """Test clustering with too few segments."""
    extractor = ThemeExtractor()

    # Create only 2 segments
    segments = []
    for i in range(2):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.text = f"Text {i}"
        seg.embedding = np.random.randn(1024)
        segments.append(seg)

    themes = extractor.extract_by_clustering(
        segments=segments,
        min_cluster_size=5
    )

    # Should return empty list
    assert len(themes) == 0


def test_extract_by_queries(mock_segments, mock_llm_service):
    """Test query-based extraction."""
    extractor = ThemeExtractor(llm_service=mock_llm_service)

    # Mock query embeddings to match segment groups
    query_embeddings = [
        np.array([1.0] * 512 + [0.0] * 512),  # Politics
        np.array([0.0] * 512 + [1.0] * 512),  # Economy
    ]
    mock_llm_service.batch_convert_queries_to_embeddings.return_value = query_embeddings

    themes = extractor.extract_by_queries(
        segments=mock_segments,
        theme_queries=["politics", "economy"],
        threshold=0.5
    )

    # Should find themes matching queries
    assert len(themes) >= 1
    assert all(isinstance(theme, Theme) for theme in themes)
    assert all(theme.metadata.get("query") in ["politics", "economy"] for theme in themes)


def test_extract_by_queries_no_llm():
    """Test query-based extraction without LLM."""
    extractor = ThemeExtractor()

    with pytest.raises(ValueError, match="LLMService required"):
        extractor.extract_by_queries(
            segments=[],
            theme_queries=["test"]
        )


def test_extract_by_keywords_text(mock_segments):
    """Test text-based keyword extraction."""
    extractor = ThemeExtractor()

    themes = extractor.extract_by_keywords(
        segments=mock_segments,
        keywords=["political", "economic"],
        use_embeddings=False,
        min_segments_per_theme=3
    )

    # Should find themes based on keyword matching
    assert len(themes) >= 1
    assert all(isinstance(theme, Theme) for theme in themes)


def test_extract_by_keywords_semantic(mock_segments, mock_llm_service):
    """Test semantic keyword extraction."""
    extractor = ThemeExtractor(llm_service=mock_llm_service)

    # Mock to use extract_by_queries
    with patch.object(extractor, 'extract_by_queries') as mock_extract:
        mock_extract.return_value = []

        extractor.extract_by_keywords(
            segments=mock_segments,
            keywords=["politics", "economy"],
            use_embeddings=True
        )

        # Should call extract_by_queries
        mock_extract.assert_called_once()


def test_align_themes_across_groups():
    """Test theme alignment across groups."""
    extractor = ThemeExtractor()

    # Create similar themes across groups
    embedding1 = np.random.randn(1024)
    embedding2 = np.random.randn(1024)

    theme_en_1 = Theme(
        theme_id="en_1",
        theme_name="Politics EN",
        segments=[],
        representative_segments=[],
        embedding=embedding1
    )

    theme_fr_1 = Theme(
        theme_id="fr_1",
        theme_name="Politique FR",
        segments=[],
        representative_segments=[],
        embedding=embedding1 + np.random.randn(1024) * 0.01  # Very similar
    )

    theme_de_1 = Theme(
        theme_id="de_1",
        theme_name="Different topic",
        segments=[],
        representative_segments=[],
        embedding=embedding2  # Different
    )

    group_themes = {
        "en": [theme_en_1],
        "fr": [theme_fr_1],
        "de": [theme_de_1]
    }

    aligned = extractor.align_themes_across_groups(
        group_themes=group_themes,
        alignment_threshold=0.9
    )

    # EN and FR should be aligned, DE separate
    assert len(aligned) == 3
    en_themes = aligned["en"]
    fr_themes = aligned["fr"]

    # Check alignment metadata
    assert en_themes[0].metadata.get("aligned") is not None
    assert fr_themes[0].metadata.get("aligned") is not None


def test_extract_subthemes(mock_segments):
    """Test sub-theme extraction."""
    extractor = ThemeExtractor()

    # Create parent theme
    parent_theme = Theme(
        theme_id="parent",
        theme_name="Parent Theme",
        segments=mock_segments[:20],  # Use 20 segments
        representative_segments=mock_segments[:5]
    )

    subthemes = extractor.extract_subthemes(
        theme=parent_theme,
        method="hdbscan",
        min_cluster_size=3
    )

    # Should find sub-themes
    assert len(subthemes) >= 1
    assert all(sub.parent_theme_id == "parent" for sub in subthemes)
    assert all(sub.depth == 1 for sub in subthemes)
    assert all(sub.theme_id.startswith("parent_sub_") for sub in subthemes)


def test_extract_subthemes_too_few_segments():
    """Test sub-theme extraction with too few segments."""
    extractor = ThemeExtractor()

    # Create parent theme with only 5 segments
    segments = []
    for i in range(5):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.text = f"Text {i}"
        seg.embedding = np.random.randn(1024)
        segments.append(seg)

    parent_theme = Theme(
        theme_id="parent",
        theme_name="Parent Theme",
        segments=segments,
        representative_segments=segments[:2]
    )

    subthemes = extractor.extract_subthemes(
        theme=parent_theme,
        min_cluster_size=3
    )

    # Should return empty list
    assert len(subthemes) == 0


@pytest.mark.asyncio
async def test_extract_subthemes_batch(mock_segments):
    """Test batch sub-theme extraction."""
    extractor = ThemeExtractor()

    # Create parent themes
    parent_themes = [
        Theme(
            theme_id=f"parent_{i}",
            theme_name=f"Parent Theme {i}",
            segments=mock_segments[i*10:(i+1)*10] if i < 2 else mock_segments[20:25],
            representative_segments=mock_segments[i*10:i*10+3]
        )
        for i in range(2)
    ]

    subtheme_map = await extractor.extract_subthemes_batch(
        themes=parent_themes,
        method="hdbscan",
        min_cluster_size=3,
        max_concurrent=2
    )

    # Should have subthemes for each parent
    assert len(subtheme_map) == 2
    assert all(theme_id in subtheme_map for theme_id in ["parent_0", "parent_1"])


def test_theme_representative_text():
    """Test Theme.representative_text property."""
    segments = []
    for i in range(10):
        seg = Mock(spec=Segment)
        seg.text = f"Text {i}"
        segments.append(seg)

    theme = Theme(
        theme_id="test",
        theme_name="Test Theme",
        segments=segments,
        representative_segments=segments[:5]
    )

    rep_text = theme.representative_text
    assert rep_text == "Text 0 Text 1 Text 2 Text 3 Text 4"


def test_theme_len():
    """Test Theme.__len__."""
    segments = [Mock(spec=Segment) for _ in range(15)]

    theme = Theme(
        theme_id="test",
        theme_name="Test Theme",
        segments=segments,
        representative_segments=segments[:5]
    )

    assert len(theme) == 15


def test_select_representative_segments(mock_segments):
    """Test representative segment selection."""
    extractor = ThemeExtractor()

    # Use first 15 segments
    segments = mock_segments[:15]
    embeddings = np.vstack([seg.embedding for seg in segments])

    rep_segments = extractor._select_representative_segments(
        segments=segments,
        embeddings=embeddings,
        n=5
    )

    # Should return 5 most central segments
    assert len(rep_segments) == 5
    assert all(seg in segments for seg in rep_segments)


def test_select_representative_segments_fewer_than_n(mock_segments):
    """Test representative selection when fewer than n segments."""
    extractor = ThemeExtractor()

    segments = mock_segments[:3]
    embeddings = np.vstack([seg.embedding for seg in segments])

    rep_segments = extractor._select_representative_segments(
        segments=segments,
        embeddings=embeddings,
        n=10
    )

    # Should return all segments
    assert len(rep_segments) == 3
    assert rep_segments == segments


# ============================================================================
# Cluster Validation Tests
# ============================================================================

def test_validate_clusters_valid():
    """Test cluster validation with well-separated clusters."""
    extractor = ThemeExtractor()

    # Create clearly separated clusters
    embeddings = np.vstack([
        np.random.randn(10, 1024) + np.array([5.0] * 512 + [0.0] * 512),  # Cluster 0
        np.random.randn(10, 1024) + np.array([0.0] * 512 + [5.0] * 512),  # Cluster 1
    ])
    labels = np.array([0] * 10 + [1] * 10)

    is_valid, metrics = extractor._validate_clusters(embeddings, labels, min_silhouette_score=0.15)

    # Should be valid with good silhouette score
    assert is_valid is True
    assert metrics["is_valid"] is True
    assert metrics["num_clusters"] == 2
    assert metrics["silhouette_score"] > 0.15
    assert "davies_bouldin_index" in metrics
    assert "calinski_harabasz_score" in metrics


def test_validate_clusters_invalid():
    """Test cluster validation with poorly separated clusters."""
    extractor = ThemeExtractor()

    # Create overlapping clusters (poor separation)
    embeddings = np.random.randn(20, 1024)  # All similar
    labels = np.array([0] * 10 + [1] * 10)  # Arbitrary split

    is_valid, metrics = extractor._validate_clusters(embeddings, labels, min_silhouette_score=0.15)

    # Should be invalid with low silhouette score
    assert is_valid is False
    assert metrics["is_valid"] is False
    assert metrics["silhouette_score"] < 0.15
    assert metrics["reason"] == "low_silhouette_score"


def test_validate_clusters_insufficient_clusters():
    """Test validation with only 1 cluster."""
    extractor = ThemeExtractor()

    embeddings = np.random.randn(20, 1024)
    labels = np.array([0] * 20)  # All same cluster

    is_valid, metrics = extractor._validate_clusters(embeddings, labels)

    # Should be invalid - need at least 2 clusters
    assert is_valid is False
    assert metrics["is_valid"] is False
    assert metrics["num_clusters"] == 1
    assert metrics["reason"] == "insufficient_clusters_or_points"


def test_validate_clusters_with_noise():
    """Test validation with noise points."""
    extractor = ThemeExtractor()

    # Create clusters with noise points
    embeddings = np.vstack([
        np.random.randn(10, 1024) + np.array([5.0] * 512 + [0.0] * 512),  # Cluster 0
        np.random.randn(10, 1024) + np.array([0.0] * 512 + [5.0] * 512),  # Cluster 1
        np.random.randn(5, 1024),  # Noise
    ])
    labels = np.array([0] * 10 + [1] * 10 + [-1] * 5)

    is_valid, metrics = extractor._validate_clusters(embeddings, labels, min_silhouette_score=0.15)

    # Should still be valid, noise filtered out
    assert metrics["num_clusters"] == 2
    assert metrics["num_noise_points"] == 5
    assert metrics["total_points"] == 25


def test_validate_clusters_too_few_points():
    """Test validation with too few points."""
    extractor = ThemeExtractor()

    # Only 5 points across 2 clusters
    embeddings = np.random.randn(5, 1024)
    labels = np.array([0, 0, 1, 1, 1])

    is_valid, metrics = extractor._validate_clusters(embeddings, labels)

    # Should be invalid - need at least 10 points
    assert is_valid is False
    assert metrics["reason"] == "insufficient_clusters_or_points"


def test_extract_subthemes_with_validation_valid():
    """Test sub-theme extraction with valid clusters."""
    extractor = ThemeExtractor()

    # Create parent theme with clearly separated sub-groups
    segments = []
    for i in range(20):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.text = f"Text {i}"
        # Create two distinct groups
        if i < 10:
            seg.embedding = np.random.randn(1024) + np.array([5.0] * 512 + [0.0] * 512)
        else:
            seg.embedding = np.random.randn(1024) + np.array([0.0] * 512 + [5.0] * 512)
        segments.append(seg)

    parent_theme = Theme(
        theme_id="parent",
        theme_name="Parent Theme",
        segments=segments,
        representative_segments=segments[:5]
    )

    subthemes = extractor.extract_subthemes(
        theme=parent_theme,
        method="kmeans",
        n_subthemes=2,
        min_cluster_size=3,
        require_valid_clusters=True,
        min_silhouette_score=0.15
    )

    # Should return sub-themes with validation metadata
    assert len(subthemes) >= 1
    for subtheme in subthemes:
        assert "cluster_validation" in subtheme.metadata
        assert subtheme.metadata["cluster_validation"]["is_valid"] is True
        assert subtheme.metadata["cluster_validation"]["silhouette_score"] > 0.15


def test_extract_subthemes_with_validation_invalid():
    """Test sub-theme extraction with invalid clusters (homogeneous segments)."""
    extractor = ThemeExtractor()

    # Create parent theme with homogeneous segments (all similar)
    segments = []
    base_embedding = np.random.randn(1024)
    for i in range(20):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.text = f"Similar text {i}"
        # All embeddings are very similar
        seg.embedding = base_embedding + np.random.randn(1024) * 0.01
        segments.append(seg)

    parent_theme = Theme(
        theme_id="parent",
        theme_name="Parent Theme",
        segments=segments,
        representative_segments=segments[:5]
    )

    subthemes = extractor.extract_subthemes(
        theme=parent_theme,
        method="kmeans",
        n_subthemes=2,
        min_cluster_size=3,
        require_valid_clusters=True,
        min_silhouette_score=0.15
    )

    # Should return empty list due to poor cluster quality
    assert len(subthemes) == 0


def test_extract_subthemes_validation_disabled():
    """Test sub-theme extraction with validation disabled."""
    extractor = ThemeExtractor()

    # Create parent theme with homogeneous segments
    segments = []
    base_embedding = np.random.randn(1024)
    for i in range(20):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.text = f"Similar text {i}"
        seg.embedding = base_embedding + np.random.randn(1024) * 0.01
        segments.append(seg)

    parent_theme = Theme(
        theme_id="parent",
        theme_name="Parent Theme",
        segments=segments,
        representative_segments=segments[:5]
    )

    subthemes = extractor.extract_subthemes(
        theme=parent_theme,
        method="kmeans",
        n_subthemes=2,
        min_cluster_size=3,
        require_valid_clusters=False  # Validation disabled
    )

    # Should return sub-themes even with poor quality
    assert len(subthemes) >= 1


@pytest.mark.asyncio
async def test_extract_subthemes_batch_with_validation():
    """Test batch sub-theme extraction with validation."""
    extractor = ThemeExtractor()

    # Create parent themes: one with distinct sub-groups, one homogeneous
    # Theme 1: Well-separated
    segments1 = []
    for i in range(20):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.text = f"Text {i}"
        if i < 10:
            seg.embedding = np.random.randn(1024) + np.array([5.0] * 512 + [0.0] * 512)
        else:
            seg.embedding = np.random.randn(1024) + np.array([0.0] * 512 + [5.0] * 512)
        segments1.append(seg)

    # Theme 2: Homogeneous
    segments2 = []
    base_embedding = np.random.randn(1024)
    for i in range(20, 40):
        seg = Mock(spec=Segment)
        seg.id = i
        seg.text = f"Similar text {i}"
        seg.embedding = base_embedding + np.random.randn(1024) * 0.01
        segments2.append(seg)

    parent_themes = [
        Theme(
            theme_id="parent_0",
            theme_name="Distinct Theme",
            segments=segments1,
            representative_segments=segments1[:5]
        ),
        Theme(
            theme_id="parent_1",
            theme_name="Homogeneous Theme",
            segments=segments2,
            representative_segments=segments2[:5]
        )
    ]

    subtheme_map = await extractor.extract_subthemes_batch(
        themes=parent_themes,
        method="kmeans",
        n_subthemes_per_theme=2,
        min_cluster_size=3,
        require_valid_clusters=True,
        min_silhouette_score=0.15
    )

    # Should have entries for both themes
    assert len(subtheme_map) == 2
    assert "parent_0" in subtheme_map
    assert "parent_1" in subtheme_map

    # Theme 0 should have sub-themes (well-separated)
    assert len(subtheme_map["parent_0"]) >= 1

    # Theme 1 might have empty list (homogeneous, fails validation)
    # (could be 0 or more depending on random seed)
