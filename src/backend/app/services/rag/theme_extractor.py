"""
Theme extraction component for RAG system.

Supports multiple extraction strategies:
- Clustering (HDBSCAN, k-means, agglomerative)
- Query-based (semantic matching)
- Keyword-based (text or semantic)
- Hierarchical sub-themes (recursive clustering)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set, Tuple
import numpy as np
import warnings

# Suppress known harmless warnings
warnings.filterwarnings('ignore', message='n_jobs value .* overridden')
warnings.filterwarnings('ignore', message="'force_all_finite' was renamed")

import umap
import hdbscan
import faiss
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
from collections import defaultdict

from ...models.db_models import EmbeddingSegment as Segment
from ..llm_service import LLMService


def _get_segment_embedding(seg):
    """Get segment embedding (handles both dict and object segments)."""
    if isinstance(seg, dict):
        # Check _embedding first (transient field from search), then embedding
        emb = seg.get("_embedding") or seg.get("embedding")
        if emb is not None and not isinstance(emb, np.ndarray):
            return np.array(emb)
        return emb
    return seg.embedding


def _get_segment_id(seg):
    """Get segment ID (handles both dict and object segments)."""
    if isinstance(seg, dict):
        return seg.get("segment_id")
    return seg.id


def _get_segment_text(seg):
    """Get segment text (handles both dict and object segments)."""
    if isinstance(seg, dict):
        return seg.get("text", "")
    return seg.text


from ..embedding_service import EmbeddingService
from ...utils.backend_logger import get_logger

logger = get_logger("theme_extractor")


@dataclass
class Theme:
    """Represents a discovered theme/topic."""
    theme_id: str
    theme_name: str
    segments: List[Segment]
    representative_segments: List[Segment]  # Top N most representative
    keywords: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Hierarchical support
    parent_theme_id: Optional[str] = None  # For sub-themes
    subthemes: Optional[List["Theme"]] = None  # Nested sub-themes
    depth: int = 0  # 0 = top-level, 1 = sub-theme, etc.

    # Selected segments for summarization (set by select_segments step)
    selected: List[Segment] = field(default_factory=list)

    @property
    def representative_text(self) -> str:
        """Get concatenated text of representative segments."""
        return " ".join([seg.text for seg in self.representative_segments[:5]])

    def __len__(self) -> int:
        return len(self.segments)


class ThemeExtractor:
    """Extract themes from segments using various strategies."""

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        embedding_service: Optional[EmbeddingService] = None
    ):
        self.llm = llm_service
        self.embedding_service = embedding_service

    def extract_by_clustering(
        self,
        segments: List[Segment],
        method: str = "hdbscan",
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 5,
        umap_params: Optional[Dict] = None,
        min_segments_per_theme: int = 3,
        max_themes: Optional[int] = None,
        use_faiss: bool = False,
        min_theme_percentage: Optional[float] = None
    ) -> List[Theme]:
        """
        Extract themes using clustering.

        Args:
            segments: Segments to cluster
            method: Clustering method (hdbscan, kmeans, agglomerative)
            n_clusters: Number of clusters (required for kmeans/agglomerative)
            min_cluster_size: Minimum cluster size for HDBSCAN
            umap_params: UMAP dimensionality reduction parameters
            min_segments_per_theme: Minimum segments to form a valid theme
            max_themes: Maximum number of themes to return
            use_faiss: Use FAISS for accelerated clustering on large datasets
            min_theme_percentage: Minimum percentage of total segments for a theme (e.g., 5.0 for 5%)

        Returns:
            List of extracted themes
        """
        if len(segments) < min_segments_per_theme:
            logger.warning(f"Too few segments ({len(segments)}) for clustering")
            return []

        # Get embeddings (handle both dict and object segments)
        embeddings = np.vstack([_get_segment_embedding(seg) for seg in segments if _get_segment_embedding(seg) is not None])
        valid_segments = [seg for seg in segments if _get_segment_embedding(seg) is not None]

        if len(valid_segments) < min_segments_per_theme:
            logger.warning("Too few segments with embeddings")
            return []

        total_segments = len(valid_segments)
        logger.info(f"Clustering {total_segments} segments (use_faiss={use_faiss})")

        # Build FAISS index if requested (for large datasets)
        faiss_index = None
        if use_faiss and len(valid_segments) > 1000:
            logger.info("Building FAISS index for accelerated clustering...")
            # Normalize embeddings for cosine similarity
            normalized_embeddings = embeddings.copy()
            faiss.normalize_L2(normalized_embeddings)

            # Build IVFFlat index for fast search
            d = embeddings.shape[1]  # Dimension
            nlist = min(int(np.sqrt(len(valid_segments))), 100)  # Number of clusters for IVF
            quantizer = faiss.IndexFlatIP(d)  # Inner product = cosine after normalization
            faiss_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

            # Train and add vectors
            faiss_index.train(normalized_embeddings.astype('float32'))
            faiss_index.add(normalized_embeddings.astype('float32'))
            logger.info(f"FAISS index built: {nlist} clusters, {len(valid_segments)} vectors")

        # UMAP dimensionality reduction
        umap_params = umap_params or {
            "n_neighbors": min(15, len(valid_segments) - 1),
            "n_components": min(5, len(valid_segments) - 1),
            "metric": "cosine",
            "random_state": 42
        }

        logger.info(f"Running UMAP with params: {umap_params}")
        reducer = umap.UMAP(**umap_params)
        reduced_embeddings = reducer.fit_transform(embeddings)

        # Clustering
        if method == "hdbscan":
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                metric="euclidean",
                cluster_selection_method="eom"
            )
            cluster_labels = clusterer.fit_predict(reduced_embeddings)
        elif method == "kmeans":
            if n_clusters is None:
                raise ValueError("n_clusters required for kmeans")
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(reduced_embeddings)
        elif method == "agglomerative":
            if n_clusters is None:
                raise ValueError("n_clusters required for agglomerative")
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            cluster_labels = clusterer.fit_predict(reduced_embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Build themes from clusters
        themes = self._build_themes_from_clusters(
            valid_segments,
            embeddings,
            cluster_labels,
            min_segments_per_theme
        )

        # Apply volume threshold filter if specified
        if min_theme_percentage is not None and min_theme_percentage > 0:
            min_theme_size = int(total_segments * (min_theme_percentage / 100.0))
            before_filter = len(themes)
            themes = [t for t in themes if len(t.segments) >= min_theme_size]
            if before_filter > len(themes):
                logger.info(f"Volume threshold filter: {before_filter} → {len(themes)} themes (min {min_theme_size} segments = {min_theme_percentage}%)")

        # Sort by size and limit to max_themes
        themes = sorted(themes, key=len, reverse=True)
        if max_themes and len(themes) > max_themes:
            logger.info(f"Limiting to top {max_themes} themes (from {len(themes)})")
            themes = themes[:max_themes]

        logger.info(f"Extracted {len(themes)} themes from {total_segments} segments")
        return themes

    def extract_with_forced_kmeans(
        self,
        segments: List[Segment],
        reduced_embeddings: np.ndarray,
        k_values: List[int] = None,
        min_silhouette: float = 0.15,
        min_segments_per_theme: int = 10
    ) -> Tuple[Optional[List[Theme]], Dict[str, Any]]:
        """
        Try forcing k-means clustering with multiple k values and validate with silhouette.

        Args:
            segments: Segments with embeddings
            reduced_embeddings: UMAP-reduced embeddings (already computed)
            k_values: List of k values to try (default: [2, 3, 4])
            min_silhouette: Minimum silhouette score to accept clustering
            min_segments_per_theme: Minimum segments per theme

        Returns:
            Tuple of (best themes or None, validation metrics dict)
        """
        k_values = k_values or [2, 3, 4]
        best_themes = None
        best_k = None
        best_silhouette = -1
        best_metrics = {}

        logger.info(f"Trying k-means fallback with k={k_values}")

        for k in k_values:
            # Skip if k is too large for the data
            if k > len(segments) // min_segments_per_theme:
                continue

            # Run k-means
            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(reduced_embeddings)

            # Build themes from clusters
            themes = self._build_themes_from_clusters(
                segments,
                np.vstack([_get_segment_embedding(seg) for seg in segments]),
                cluster_labels,
                min_segments_per_theme
            )

            # Skip if we didn't get k themes (some clusters too small)
            if len(themes) < 2:
                logger.debug(f"k={k}: Only {len(themes)} valid themes after filtering")
                continue

            # Validate with silhouette score
            embeddings = np.vstack([_get_segment_embedding(seg) for seg in segments if _get_segment_embedding(seg) is not None])
            is_valid, metrics = self._validate_clusters(
                embeddings,
                cluster_labels,
                min_silhouette_score=min_silhouette
            )

            silhouette = metrics.get("silhouette_score", -1)
            logger.info(f"k={k}: {len(themes)} themes, silhouette={silhouette:.3f}, valid={is_valid}")

            # Track best result
            if is_valid and silhouette > best_silhouette:
                best_silhouette = silhouette
                best_themes = themes
                best_k = k
                best_metrics = metrics

        if best_themes:
            logger.info(f"✓ K-means fallback successful: k={best_k}, silhouette={best_silhouette:.3f}")
            return best_themes, best_metrics
        else:
            logger.info(f"✗ K-means fallback failed: no valid clustering found")
            return None, {"silhouette_score": best_silhouette, "tried_k_values": k_values}

    def extract_by_queries(
        self,
        segments: List[Segment],
        theme_queries: List[str],
        threshold: float = 0.75,
        min_segments_per_theme: int = 3
    ) -> List[Theme]:
        """
        Extract themes by matching predefined queries.

        Args:
            segments: Segments to match
            theme_queries: List of query strings defining themes
            threshold: Minimum similarity threshold
            min_segments_per_theme: Minimum segments per theme

        Returns:
            List of themes
        """
        if not self.embedding_service:
            raise ValueError("EmbeddingService required for query-based extraction")

        # Get query embeddings synchronously using pre-loaded model
        query_embeddings_list = [
            self.embedding_service.encode_query(q) for q in theme_queries
        ]
        query_embeddings = np.array(query_embeddings_list)

        # Get segment embeddings
        segment_embeddings = np.vstack([_get_segment_embedding(seg) for seg in segments if _get_segment_embedding(seg) is not None])
        valid_segments = [seg for seg in segments if _get_segment_embedding(seg) is not None]

        # Compute similarities
        similarities = cosine_similarity(query_embeddings, segment_embeddings)

        # Assign segments to themes
        themes = []
        for i, query in enumerate(theme_queries):
            # Find segments above threshold
            matching_indices = np.where(similarities[i] >= threshold)[0]

            if len(matching_indices) >= min_segments_per_theme:
                theme_segments = [valid_segments[idx] for idx in matching_indices]
                theme = Theme(
                    theme_id=f"query_theme_{i}",
                    theme_name=query,
                    segments=theme_segments,
                    representative_segments=self._select_representative_segments(
                        theme_segments,
                        segment_embeddings[matching_indices]
                    ),
                    metadata={"query": query, "threshold": threshold}
                )
                themes.append(theme)

        logger.info(f"Extracted {len(themes)} themes from {len(theme_queries)} queries")
        return themes

    def extract_by_keywords(
        self,
        segments: List[Segment],
        keywords: List[str],
        use_embeddings: bool = True,
        threshold: float = 0.7,
        min_segments_per_theme: int = 3
    ) -> List[Theme]:
        """
        Extract themes by keyword matching.

        Args:
            segments: Segments to match
            keywords: List of keywords
            use_embeddings: Use semantic matching vs text matching
            threshold: Similarity threshold (for semantic matching)
            min_segments_per_theme: Minimum segments per theme

        Returns:
            List of themes
        """
        if use_embeddings:
            if not self.llm:
                raise ValueError("LLMService required for semantic keyword matching")
            return self.extract_by_queries(segments, keywords, threshold, min_segments_per_theme)

        # Text-based keyword matching
        themes = []
        for keyword in keywords:
            keyword_lower = keyword.lower()
            matching_segments = [
                seg for seg in segments
                if keyword_lower in _get_segment_text(seg).lower()
            ]

            if len(matching_segments) >= min_segments_per_theme:
                embeddings = np.vstack([_get_segment_embedding(seg) for seg in matching_segments if _get_segment_embedding(seg) is not None])
                theme = Theme(
                    theme_id=f"keyword_theme_{keyword}",
                    theme_name=keyword,
                    segments=matching_segments,
                    representative_segments=self._select_representative_segments(
                        matching_segments,
                        embeddings
                    ),
                    keywords=[keyword],
                    metadata={"keyword": keyword, "method": "text_match"}
                )
                themes.append(theme)

        logger.info(f"Extracted {len(themes)} themes from {len(keywords)} keywords")
        return themes

    def align_themes_across_groups(
        self,
        group_themes: Dict[str, List[Theme]],
        alignment_threshold: float = 0.8
    ) -> Dict[str, List[Theme]]:
        """
        Align themes across groups (e.g., EN/FR/DE).

        Finds similar themes across groups and ensures consistent theme IDs.

        Args:
            group_themes: Dict mapping group_id -> list of themes
            alignment_threshold: Minimum similarity to consider themes aligned

        Returns:
            Dict mapping group_id -> aligned themes
        """
        # Extract all theme embeddings
        all_themes = []
        for group_id, themes in group_themes.items():
            for theme in themes:
                all_themes.append((group_id, theme))

        # Compute pairwise similarities
        embeddings = []
        for _, theme in all_themes:
            if theme.embedding is not None:
                embeddings.append(theme.embedding)
            else:
                # Use mean of segment embeddings
                seg_embeds = [_get_segment_embedding(seg) for seg in theme.segments if _get_segment_embedding(seg) is not None]
                if seg_embeds:
                    embeddings.append(np.mean(seg_embeds, axis=0))
                else:
                    embeddings.append(np.zeros(1024))  # Placeholder

        embeddings = np.vstack(embeddings)
        similarities = cosine_similarity(embeddings)

        # Find aligned theme groups
        aligned_groups = []
        used_indices = set()

        for i in range(len(all_themes)):
            if i in used_indices:
                continue

            # Find all themes similar to this one
            similar_indices = np.where(similarities[i] >= alignment_threshold)[0]
            aligned_group = [all_themes[idx] for idx in similar_indices if idx not in used_indices]

            if len(aligned_group) > 1:  # Cross-group alignment
                aligned_groups.append(aligned_group)
                used_indices.update(similar_indices)

        # Assign consistent theme IDs
        aligned_result = {group_id: [] for group_id in group_themes.keys()}

        for align_idx, aligned_group in enumerate(aligned_groups):
            aligned_theme_id = f"aligned_theme_{align_idx}"

            for group_id, theme in aligned_group:
                theme.theme_id = aligned_theme_id
                theme.metadata["aligned"] = True
                theme.metadata["alignment_group_size"] = len(aligned_group)
                aligned_result[group_id].append(theme)

        # Add unaligned themes
        for group_id, themes in group_themes.items():
            for theme in themes:
                if not theme.metadata.get("aligned"):
                    theme.metadata["aligned"] = False
                    aligned_result[group_id].append(theme)

        logger.info(f"Aligned {len(aligned_groups)} theme groups across {len(group_themes)} groups")
        return aligned_result

    def extract_subthemes(
        self,
        theme: Theme,
        method: str = "hdbscan",
        n_subthemes: Optional[int] = None,
        min_cluster_size: int = 3,
        require_valid_clusters: bool = True,
        min_silhouette_score: float = 0.15,
        min_clusters_to_validate: int = 2,
        **kwargs
    ) -> List[Theme]:
        """
        Extract sub-themes within a parent theme with adaptive cluster validation.

        Hierarchical clustering: run clustering on segments within a theme
        to identify nuanced positions/perspectives within the broader topic.

        NEW: Validates whether meaningful sub-clusters exist before returning results.
        This prevents forced splits when segments are homogeneous.

        Args:
            theme: Parent theme
            method: Clustering method
            n_subthemes: Number of sub-themes (for kmeans/agglomerative)
            min_cluster_size: Minimum cluster size
            require_valid_clusters: If True, validate clusters before returning
            min_silhouette_score: Minimum silhouette score for valid clusters (0.15-0.3 typical)
            min_clusters_to_validate: Minimum number of clusters required (default 2)
            **kwargs: Additional parameters for extract_by_clustering

        Returns:
            List of sub-themes (Theme objects with parent_theme_id set)
            Returns empty list if validation fails and require_valid_clusters=True
        """
        if len(theme.segments) < min_cluster_size * 2:
            logger.warning(f"Theme {theme.theme_id} has too few segments for sub-clustering")
            return []

        # Extract sub-themes using clustering
        subthemes = self.extract_by_clustering(
            segments=theme.segments,
            method=method,
            n_clusters=n_subthemes,
            min_cluster_size=min_cluster_size,
            min_segments_per_theme=min_cluster_size,
            **kwargs
        )

        # Validate clusters if requested
        if require_valid_clusters and len(subthemes) >= min_clusters_to_validate:
            # Get embeddings and labels for validation
            embeddings = np.vstack([_get_segment_embedding(seg) for seg in theme.segments if _get_segment_embedding(seg) is not None])

            # Build cluster labels array
            cluster_labels = np.full(len(embeddings), -1, dtype=int)  # -1 = noise
            valid_segments = [seg for seg in theme.segments if _get_segment_embedding(seg) is not None]

            for i, subtheme in enumerate(subthemes):
                subtheme_seg_ids = {_get_segment_id(seg) for seg in subtheme.segments}
                for j, seg in enumerate(valid_segments):
                    seg_id = _get_segment_id(seg)
                    if seg_id in subtheme_seg_ids:
                        cluster_labels[j] = i

            # Validate clusters
            is_valid, metrics = self._validate_clusters(embeddings, cluster_labels)

            # Store validation metrics in each subtheme
            for subtheme in subthemes:
                subtheme.metadata["cluster_validation"] = metrics

            if not is_valid:
                logger.info(
                    f"Theme {theme.theme_id}: Sub-clusters not valid "
                    f"(silhouette={metrics.get('silhouette_score', 0):.3f} < {min_silhouette_score}). "
                    f"Skipping sub-theme extraction."
                )
                return []

            logger.info(
                f"Theme {theme.theme_id}: Valid sub-clusters detected "
                f"(silhouette={metrics.get('silhouette_score', 0):.3f}, {len(subthemes)} clusters)"
            )

        # Set parent relationship
        for i, subtheme in enumerate(subthemes):
            subtheme.parent_theme_id = theme.theme_id
            subtheme.depth = theme.depth + 1
            subtheme.theme_id = f"{theme.theme_id}_sub_{i}"
            subtheme.metadata["parent_theme_name"] = theme.theme_name

        logger.info(f"Extracted {len(subthemes)} sub-themes from theme {theme.theme_id}")
        return subthemes

    async def extract_subthemes_batch(
        self,
        themes: List[Theme],
        method: str = "hdbscan",
        n_subthemes_per_theme: Optional[int] = None,
        min_cluster_size: int = 3,
        max_concurrent: int = 10,
        require_valid_clusters: bool = True,
        min_silhouette_score: float = 0.15,
        min_clusters_to_validate: int = 2
    ) -> Dict[str, List[Theme]]:
        """
        Extract sub-themes for multiple themes in parallel with validation.

        Args:
            themes: List of parent themes
            method: Clustering method
            n_subthemes_per_theme: Number of sub-themes per theme
            min_cluster_size: Minimum cluster size
            max_concurrent: Maximum concurrent extractions
            require_valid_clusters: If True, validate clusters before returning
            min_silhouette_score: Minimum silhouette score for valid clusters
            min_clusters_to_validate: Minimum number of clusters required

        Returns:
            Dict mapping parent theme_id -> list of sub-themes
            Themes with invalid clusters will have empty lists
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_one(theme: Theme) -> Tuple[str, List[Theme]]:
            async with semaphore:
                # Run in thread pool (blocking operation)
                loop = asyncio.get_event_loop()
                subthemes = await loop.run_in_executor(
                    None,
                    self.extract_subthemes,
                    theme,
                    method,
                    n_subthemes_per_theme,
                    min_cluster_size,
                    require_valid_clusters,
                    min_silhouette_score,
                    min_clusters_to_validate
                )
                return theme.theme_id, subthemes

        # Run all extractions in parallel
        results = await asyncio.gather(*[extract_one(theme) for theme in themes])

        subtheme_map = dict(results)
        total_subthemes = sum(len(subthemes) for subthemes in subtheme_map.values())
        themes_with_subthemes = sum(1 for subthemes in subtheme_map.values() if len(subthemes) > 0)
        logger.info(
            f"Extracted {total_subthemes} total sub-themes from {themes_with_subthemes}/{len(themes)} themes "
            f"(validation={'enabled' if require_valid_clusters else 'disabled'})"
        )

        return subtheme_map

    def _build_themes_from_clusters(
        self,
        segments: List[Segment],
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        min_segments_per_theme: int
    ) -> List[Theme]:
        """Build Theme objects from cluster labels."""
        themes = []
        unique_labels = set(cluster_labels)

        # Remove noise label if present
        if -1 in unique_labels:
            unique_labels.remove(-1)

        for label in unique_labels:
            indices = np.where(cluster_labels == label)[0]

            if len(indices) < min_segments_per_theme:
                continue

            theme_segments = [segments[i] for i in indices]
            theme_embeddings = embeddings[indices]

            # Generate theme name (placeholder)
            theme_name = f"Theme {label}"

            theme = Theme(
                theme_id=f"theme_{label}",
                theme_name=theme_name,
                segments=theme_segments,
                representative_segments=self._select_representative_segments(
                    theme_segments,
                    theme_embeddings
                ),
                metadata={
                    "cluster_label": int(label),
                    "cluster_size": len(indices)
                }
            )

            themes.append(theme)

        return themes

    def _select_representative_segments(
        self,
        segments: List[Segment],
        embeddings: np.ndarray,
        n: int = 10
    ) -> List[Segment]:
        """Select most representative segments (closest to centroid)."""
        if len(segments) <= n:
            return segments

        # Compute centroid
        centroid = np.mean(embeddings, axis=0)

        # Find closest segments to centroid
        similarities = cosine_similarity([centroid], embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:n]

        return [segments[i] for i in top_indices]

    def _validate_clusters(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        min_silhouette_score: float = 0.15
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate cluster quality using multiple metrics.

        Args:
            embeddings: Segment embeddings (n_samples, n_features)
            cluster_labels: Cluster assignments (n_samples,)
            min_silhouette_score: Minimum silhouette score for validity

        Returns:
            Tuple of (is_valid, metrics_dict)
            - is_valid: Whether clusters meet quality threshold
            - metrics_dict: Dict with silhouette_score, davies_bouldin_index, num_clusters, etc.
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

        # Filter out noise points (label -1)
        mask = cluster_labels >= 0
        filtered_embeddings = embeddings[mask]
        filtered_labels = cluster_labels[mask]

        unique_labels = np.unique(filtered_labels)
        num_clusters = len(unique_labels)

        metrics = {
            "num_clusters": int(num_clusters),
            "num_noise_points": int(np.sum(~mask)),
            "total_points": int(len(cluster_labels))
        }

        # Need at least 2 clusters and enough points for validation
        if num_clusters < 2 or len(filtered_embeddings) < 10:
            logger.debug(f"Cannot validate: {num_clusters} clusters, {len(filtered_embeddings)} points")
            metrics.update({
                "silhouette_score": 0.0,
                "davies_bouldin_index": float('inf'),
                "calinski_harabasz_score": 0.0,
                "is_valid": False,
                "reason": "insufficient_clusters_or_points"
            })
            return False, metrics

        try:
            # Silhouette Score: [-1, 1], higher is better
            # > 0.3: Clear structure
            # 0.15-0.3: Weak but detectable structure
            # < 0.15: No meaningful structure
            silhouette = silhouette_score(filtered_embeddings, filtered_labels, metric='cosine')

            # Davies-Bouldin Index: [0, inf], lower is better
            # < 1.0: Good separation
            # 1.0-2.0: Moderate separation
            # > 2.0: Poor separation
            davies_bouldin = davies_bouldin_score(filtered_embeddings, filtered_labels)

            # Calinski-Harabasz Score: [0, inf], higher is better
            # Measures ratio of between-cluster to within-cluster variance
            calinski_harabasz = calinski_harabasz_score(filtered_embeddings, filtered_labels)

            metrics.update({
                "silhouette_score": float(silhouette),
                "davies_bouldin_index": float(davies_bouldin),
                "calinski_harabasz_score": float(calinski_harabasz)
            })

            # Validation criteria: silhouette score is primary metric
            is_valid = silhouette >= min_silhouette_score

            metrics["is_valid"] = is_valid
            if not is_valid:
                metrics["reason"] = "low_silhouette_score"

            return is_valid, metrics

        except Exception as e:
            logger.error(f"Error computing cluster validation metrics: {e}", exc_info=True)
            metrics.update({
                "silhouette_score": 0.0,
                "davies_bouldin_index": float('inf'),
                "calinski_harabasz_score": 0.0,
                "is_valid": False,
                "reason": "computation_error",
                "error": str(e)
            })
            return False, metrics
