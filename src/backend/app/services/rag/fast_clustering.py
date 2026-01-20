"""
Fast Clustering for Sub-theme Detection
========================================

Provides fast alternatives to UMAP + HDBSCAN for sub-theme detection.

Key optimizations:
1. PCA instead of UMAP (10-100x faster)
2. MiniBatchKMeans for fast clustering
3. Silhouette sampling for validation (not full computation)
4. Optional: FAISS for approximate nearest neighbor clustering

Typical performance:
- 100 segments: <50ms
- 500 segments: <100ms
- 2000 segments: <300ms

Compared to UMAP + HDBSCAN:
- 100 segments: ~500ms
- 500 segments: ~2s
- 2000 segments: ~10s
"""

import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

from ...utils.backend_logger import get_logger
logger = get_logger("fast_clustering")


@dataclass
class ClusterResult:
    """Result of fast clustering."""
    labels: np.ndarray           # Cluster labels per segment
    n_clusters: int              # Number of clusters found
    silhouette_score: float      # Overall silhouette score
    is_valid: bool               # Whether clusters are meaningful
    cluster_sizes: List[int]     # Size of each cluster
    reduced_embeddings: Optional[np.ndarray] = None  # PCA-reduced embeddings
    metadata: Dict[str, Any] = None


def fast_cluster(
    embeddings: np.ndarray,
    method: str = "pca_kmeans",
    n_clusters: Optional[int] = None,
    min_cluster_size: int = 10,
    min_silhouette: float = 0.15,
    pca_components: int = 32,
    max_clusters: int = 5,
    sample_silhouette: bool = True,
    silhouette_sample_size: int = 500
) -> ClusterResult:
    """
    Fast clustering without UMAP.

    Args:
        embeddings: Segment embeddings (N x dim)
        method: Clustering method
            - "pca_kmeans": PCA reduction + KMeans (fastest, good quality)
            - "pca_minibatch": PCA + MiniBatchKMeans (very fast, lower quality)
            - "direct_kmeans": KMeans on raw embeddings (accurate, slower)
            - "auto_k": Try multiple k values, pick best silhouette
        n_clusters: Number of clusters (None = auto-detect)
        min_cluster_size: Minimum segments per cluster
        min_silhouette: Minimum silhouette score for valid clustering
        pca_components: Number of PCA components
        max_clusters: Maximum clusters to try (for auto methods)
        sample_silhouette: Use sampling for silhouette (faster)
        silhouette_sample_size: Number of samples for silhouette estimation

    Returns:
        ClusterResult with labels, validity, and metrics
    """
    start_time = time.perf_counter()
    n_samples = len(embeddings)

    if n_samples < min_cluster_size * 2:
        logger.warning(f"Too few samples ({n_samples}) for clustering")
        return ClusterResult(
            labels=np.zeros(n_samples, dtype=int),
            n_clusters=1,
            silhouette_score=0.0,
            is_valid=False,
            cluster_sizes=[n_samples],
            metadata={"error": "too_few_samples"}
        )

    # Dimensionality reduction
    reduced = embeddings
    if method.startswith("pca_") or method == "auto_k":
        # PCA is MUCH faster than UMAP
        n_components = min(pca_components, min(n_samples, embeddings.shape[1]) - 1)
        pca = PCA(n_components=n_components, random_state=42)
        reduced = pca.fit_transform(embeddings)
        logger.debug(f"PCA: {embeddings.shape[1]} -> {n_components} dims, "
                    f"explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # Clustering
    if method == "auto_k":
        # Try multiple k values, pick best
        result = _auto_k_clustering(
            reduced, embeddings,
            max_k=max_clusters,
            min_cluster_size=min_cluster_size,
            min_silhouette=min_silhouette,
            sample_silhouette=sample_silhouette,
            silhouette_sample_size=silhouette_sample_size
        )
    elif n_clusters is None:
        # Default: try 2-4 clusters
        result = _auto_k_clustering(
            reduced, embeddings,
            max_k=min(4, n_samples // min_cluster_size),
            min_cluster_size=min_cluster_size,
            min_silhouette=min_silhouette,
            sample_silhouette=sample_silhouette,
            silhouette_sample_size=silhouette_sample_size
        )
    else:
        # Fixed k
        if method == "pca_minibatch":
            clusterer = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=min(256, n_samples),
                n_init=3
            )
        else:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        labels = clusterer.fit_predict(reduced)

        # Validate
        sil_score = _compute_silhouette(
            embeddings, labels,
            sample=sample_silhouette,
            sample_size=silhouette_sample_size
        )

        cluster_sizes = [int(np.sum(labels == i)) for i in range(n_clusters)]
        is_valid = sil_score >= min_silhouette and min(cluster_sizes) >= min_cluster_size

        result = ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            silhouette_score=sil_score,
            is_valid=is_valid,
            cluster_sizes=cluster_sizes,
            reduced_embeddings=reduced if method.startswith("pca_") else None
        )

    elapsed = (time.perf_counter() - start_time) * 1000
    logger.info(f"Fast clustering: {n_samples} samples -> {result.n_clusters} clusters "
               f"(sil={result.silhouette_score:.3f}, valid={result.is_valid}) in {elapsed:.1f}ms")

    result.metadata = result.metadata or {}
    result.metadata["elapsed_ms"] = elapsed
    result.metadata["method"] = method

    return result


def _auto_k_clustering(
    reduced_embeddings: np.ndarray,
    full_embeddings: np.ndarray,
    max_k: int,
    min_cluster_size: int,
    min_silhouette: float,
    sample_silhouette: bool,
    silhouette_sample_size: int
) -> ClusterResult:
    """Try multiple k values and pick the best clustering."""
    n_samples = len(reduced_embeddings)
    best_result = None
    best_score = -1

    # Try k = 2, 3, 4, ... up to max_k
    for k in range(2, max_k + 1):
        # Skip if k would create clusters smaller than min_cluster_size
        if n_samples // k < min_cluster_size:
            continue

        clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = clusterer.fit_predict(reduced_embeddings)

        # Check cluster sizes
        cluster_sizes = [int(np.sum(labels == i)) for i in range(k)]
        if min(cluster_sizes) < min_cluster_size:
            logger.debug(f"k={k}: clusters too small: {cluster_sizes}")
            continue

        # Compute silhouette on full embeddings (more accurate)
        sil_score = _compute_silhouette(
            full_embeddings, labels,
            sample=sample_silhouette,
            sample_size=silhouette_sample_size
        )

        logger.debug(f"k={k}: silhouette={sil_score:.3f}, sizes={cluster_sizes}")

        if sil_score > best_score:
            best_score = sil_score
            best_result = ClusterResult(
                labels=labels,
                n_clusters=k,
                silhouette_score=sil_score,
                is_valid=sil_score >= min_silhouette,
                cluster_sizes=cluster_sizes,
                reduced_embeddings=reduced_embeddings
            )

    if best_result is None:
        # No valid clustering found
        return ClusterResult(
            labels=np.zeros(n_samples, dtype=int),
            n_clusters=1,
            silhouette_score=0.0,
            is_valid=False,
            cluster_sizes=[n_samples],
            metadata={"error": "no_valid_clustering"}
        )

    return best_result


def _compute_silhouette(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample: bool = True,
    sample_size: int = 500
) -> float:
    """
    Compute silhouette score, optionally with sampling for speed.

    Full silhouette is O(n²) - sampling makes it O(sample_size * n).
    """
    n_samples = len(embeddings)
    unique_labels = np.unique(labels)

    # Need at least 2 clusters
    if len(unique_labels) < 2:
        return 0.0

    # For small datasets, compute full silhouette
    if n_samples <= sample_size or not sample:
        try:
            return float(silhouette_score(embeddings, labels))
        except ValueError:
            return 0.0

    # Sample for speed
    try:
        # Stratified sampling to ensure all clusters represented
        sample_indices = []
        samples_per_cluster = sample_size // len(unique_labels)

        for label in unique_labels:
            cluster_indices = np.where(labels == label)[0]
            n_take = min(samples_per_cluster, len(cluster_indices))
            if n_take > 0:
                sampled = np.random.choice(cluster_indices, size=n_take, replace=False)
                sample_indices.extend(sampled)

        sample_indices = np.array(sample_indices)
        sample_embeddings = embeddings[sample_indices]
        sample_labels = labels[sample_indices]

        return float(silhouette_score(sample_embeddings, sample_labels))
    except ValueError:
        return 0.0


def cluster_segments_fast(
    segments: List[Dict[str, Any]],
    min_cluster_size: int = 10,
    min_silhouette: float = 0.15,
    max_clusters: int = 4
) -> Tuple[List[List[Dict[str, Any]]], ClusterResult]:
    """
    Cluster segments quickly and return grouped segments.

    Args:
        segments: List of segment dicts with '_embedding' or 'embedding' field
        min_cluster_size: Minimum segments per cluster
        min_silhouette: Minimum silhouette for valid clustering
        max_clusters: Maximum clusters to try

    Returns:
        Tuple of:
        - List of segment groups (one per cluster)
        - ClusterResult with metrics
    """
    if not segments:
        return [], ClusterResult(
            labels=np.array([]),
            n_clusters=0,
            silhouette_score=0.0,
            is_valid=False,
            cluster_sizes=[],
            metadata={"error": "no_segments"}
        )

    # Extract embeddings
    embeddings_list = []
    valid_indices = []

    for i, seg in enumerate(segments):
        emb = seg.get('_embedding')
        if emb is None:
            emb = seg.get('embedding')
        if emb is not None:
            if not isinstance(emb, np.ndarray):
                emb = np.array(emb)
            embeddings_list.append(emb)
            valid_indices.append(i)

    if len(embeddings_list) < min_cluster_size * 2:
        # Not enough segments with embeddings
        return [segments], ClusterResult(
            labels=np.zeros(len(segments), dtype=int),
            n_clusters=1,
            silhouette_score=0.0,
            is_valid=False,
            cluster_sizes=[len(segments)],
            metadata={"error": "too_few_embeddings"}
        )

    embeddings = np.vstack(embeddings_list)

    # Run fast clustering
    result = fast_cluster(
        embeddings,
        method="auto_k",
        min_cluster_size=min_cluster_size,
        min_silhouette=min_silhouette,
        max_clusters=max_clusters
    )

    # Group segments by cluster
    groups = [[] for _ in range(result.n_clusters)]

    for idx, cluster_label in zip(valid_indices, result.labels):
        groups[cluster_label].append(segments[idx])

    # Add segments without embeddings to largest cluster
    segments_with_emb = set(valid_indices)
    largest_cluster = np.argmax(result.cluster_sizes)

    for i, seg in enumerate(segments):
        if i not in segments_with_emb:
            groups[largest_cluster].append(seg)

    return groups, result
