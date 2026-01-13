"""
Topic Discovery via Clustering
================================

Discovers latent topics in content through UMAP dimensionality reduction
and HDBSCAN clustering, with configurable scoring for topic ranking.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
sys.path.insert(0, str(get_project_root()))
from src.utils.logger import setup_worker_logger
logger = setup_worker_logger("backend.topic_discovery")


@dataclass
class ClusterMetrics:
    """Metrics for a single cluster"""
    cluster_id: int
    size: int
    segment_ids: List[int]

    # Scoring components
    breadth: float  # Unique channels discussing this topic
    intensity: float  # Segments per day
    coherence: float  # Cluster tightness (0-1)
    recency: float  # Time-weighted score (0-1)

    # Combined score
    score: float

    # Representative segments (closest to centroid)
    representative_segment_ids: List[int]

    # Metadata for analysis
    channels: List[str]
    date_range: Tuple[datetime, datetime]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'cluster_id': self.cluster_id,
            'size': self.size,
            'segment_ids': self.segment_ids,
            'breadth': round(self.breadth, 3),
            'intensity': round(self.intensity, 3),
            'coherence': round(self.coherence, 3),
            'recency': round(self.recency, 3),
            'score': round(self.score, 3),
            'representative_segment_ids': self.representative_segment_ids,
            'channels': self.channels,
            'date_range': (
                self.date_range[0].isoformat() if self.date_range[0] else None,
                self.date_range[1].isoformat() if self.date_range[1] else None
            )
        }


class TopicDiscovery:
    """
    Discovers topics through clustering of embedding vectors.

    Uses UMAP for dimensionality reduction followed by HDBSCAN for clustering.
    Scores topics based on breadth, intensity, coherence, and recency.
    """

    def __init__(
        self,
        search_service,
        time_window_days: int = 7,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        umap_n_components: int = 50,
        umap_n_neighbors: int = 15,
        scoring_weights: Optional[Dict[str, float]] = None,
        adaptive_sizing: bool = False,
        target_themes: int = 40,
        min_cluster_size_floor: int = 10,
        min_cluster_size_ceiling: int = 100
    ):
        """
        Initialize TopicDiscovery

        Args:
            search_service: SearchService instance (provides cached FAISS index)
            time_window_days: Time window for analysis
            min_cluster_size: Minimum cluster size for HDBSCAN (used if adaptive_sizing=False)
            min_samples: Minimum samples for HDBSCAN core points
            umap_n_components: Target dimensionality for UMAP
            umap_n_neighbors: Number of neighbors for UMAP
            scoring_weights: Dict with keys {breadth, intensity, coherence, recency}
                            Defaults: {0.35, 0.25, 0.25, 0.15}
            adaptive_sizing: If True, calculate min_cluster_size based on dataset size
            target_themes: Target number of themes (used when adaptive_sizing=True)
            min_cluster_size_floor: Minimum allowed cluster size
            min_cluster_size_ceiling: Maximum allowed cluster size
        """
        self.search_service = search_service
        self.time_window_days = time_window_days
        self.base_min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.adaptive_sizing = adaptive_sizing
        self.target_themes = target_themes
        self.min_cluster_size_floor = min_cluster_size_floor
        self.min_cluster_size_ceiling = min_cluster_size_ceiling

        # Default scoring weights (sum to 1.0)
        self.scoring_weights = scoring_weights or {
            'breadth': 0.35,      # Channel diversity is important
            'intensity': 0.25,    # Volume of discussion
            'coherence': 0.25,    # Cluster quality
            'recency': 0.15       # Time relevance
        }

        # Cached data
        self._search_index = None
        self._embeddings = None
        self._reduced_embeddings = None
        self._cluster_labels = None
        self._clusters = None

        logger.info(
            f"TopicDiscovery initialized: "
            f"window={time_window_days}d, "
            f"adaptive={adaptive_sizing}, "
            f"target_themes={target_themes if adaptive_sizing else 'N/A'}, "
            f"base_min_cluster_size={min_cluster_size}, "
            f"weights={self.scoring_weights}"
        )

    def discover_topics(self, force_rebuild: bool = False, sample_weights: Optional[np.ndarray] = None) -> List[ClusterMetrics]:
        """
        Discover topics through clustering pipeline

        Args:
            force_rebuild: Force rebuild of search index
            sample_weights: Optional per-sample weights for clustering (for balancing group sizes)

        Returns:
            List of ClusterMetrics, sorted by score (descending)
        """
        logger.info(f"Starting topic discovery for {self.time_window_days}d window")

        # Step 1: Get cached search index
        logger.info("Fetching search index...")
        self._search_index = self.search_service.get_index(
            self.time_window_days,
            force_rebuild=force_rebuild
        )

        if self._search_index.size() == 0:
            logger.warning("Empty search index, no topics to discover")
            return []

        logger.info(f"Index loaded: {self._search_index.size()} vectors ({self._search_index.dimension}-dim)")

        # Step 2: Extract embeddings from FAISS index
        logger.info("Extracting embeddings from FAISS index...")
        self._embeddings = self._extract_embeddings_from_faiss()
        logger.info(f"Extracted embeddings: {self._embeddings.shape}")

        # Step 3: Dimensionality reduction with UMAP
        logger.info(f"Running UMAP: {self._embeddings.shape[1]}d → {self.umap_n_components}d")
        self._reduced_embeddings = self._reduce_dimensions()
        logger.info(f"UMAP complete: {self._reduced_embeddings.shape}")

        # Step 4: Clustering with HDBSCAN
        logger.info("Running HDBSCAN clustering...")
        self._cluster_labels = self._cluster(sample_weights=sample_weights)

        unique_clusters = set(self._cluster_labels) - {-1}  # Exclude noise (-1)
        noise_count = np.sum(self._cluster_labels == -1)
        logger.info(
            f"HDBSCAN complete: {len(unique_clusters)} clusters, "
            f"{noise_count} noise points ({100*noise_count/len(self._cluster_labels):.1f}%)"
        )

        # Step 5: Compute metrics and score clusters
        logger.info("Computing cluster metrics and scores...")
        self._clusters = self._compute_cluster_metrics()

        # Sort by score
        self._clusters.sort(key=lambda c: c.score, reverse=True)

        logger.info(f"✓ Topic discovery complete: {len(self._clusters)} topics identified")
        if self._clusters:
            logger.info(f"  Top topic: score={self._clusters[0].score:.3f}, size={self._clusters[0].size}")

        return self._clusters

    def _extract_embeddings_from_faiss(self) -> np.ndarray:
        """Extract all embeddings from FAISS index"""
        # FAISS stores vectors in index.xb (training vectors)
        # For IndexFlatIP, all vectors are in xb
        import faiss

        n_vectors = self._search_index.index.ntotal
        dimension = self._search_index.dimension

        # Reconstruct all vectors
        embeddings = np.zeros((n_vectors, dimension), dtype=np.float32)
        for i in range(n_vectors):
            self._search_index.index.reconstruct(i, embeddings[i])

        return embeddings

    def _reduce_dimensions(self) -> np.ndarray:
        """Reduce dimensionality using UMAP"""
        import umap

        # Adaptive n_neighbors based on dataset size
        n_samples = self._embeddings.shape[0]
        n_neighbors = min(self.umap_n_neighbors, n_samples - 1)

        reducer = umap.UMAP(
            n_components=self.umap_n_components,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            metric='cosine',
            random_state=42,
            verbose=False,
            low_memory=True,  # Reduce memory usage and avoid threading issues
            n_jobs=1  # Single-threaded to avoid segfault in async context
        )

        return reducer.fit_transform(self._embeddings)

    def _cluster(self, sample_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Cluster using HDBSCAN

        Note: sample_weights parameter is kept for API compatibility but not used.
        Group balancing should be done via stratified sampling before calling this method.
        """
        import hdbscan

        # Calculate min_cluster_size
        n_samples = self._reduced_embeddings.shape[0]

        if self.adaptive_sizing:
            # Adaptive: scale with dataset size to maintain target number of themes
            # min_cluster_size ≈ n_samples / target_themes
            calculated_size = int(n_samples / self.target_themes)
            min_cluster_size = max(
                self.min_cluster_size_floor,
                min(self.min_cluster_size_ceiling, calculated_size)
            )
            logger.info(f"  Adaptive sizing: {n_samples} segments / {self.target_themes} target themes "
                       f"= {calculated_size} → clamped to {min_cluster_size}")
        else:
            # Fixed: use provided value, but clamp to reasonable range for dataset
            min_cluster_size = min(self.base_min_cluster_size, max(5, n_samples // 100))
            logger.info(f"  Fixed sizing: using min_cluster_size={min_cluster_size}")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean',
            cluster_selection_method='eom',  # Excess of mass
            core_dist_n_jobs=1  # Single-threaded to avoid segfault in async context
        )

        cluster_labels = clusterer.fit_predict(self._reduced_embeddings)

        # Store cluster probabilities for coherence scoring
        self._cluster_probabilities = clusterer.probabilities_

        return cluster_labels

    def _compute_cluster_metrics(self) -> List[ClusterMetrics]:
        """Compute metrics for each cluster"""
        metrics = []

        unique_clusters = set(self._cluster_labels) - {-1}  # Exclude noise

        for cluster_id in unique_clusters:
            # Get segments in this cluster
            mask = self._cluster_labels == cluster_id
            cluster_indices = np.where(mask)[0]
            segment_ids = [self._search_index.segment_ids[i] for i in cluster_indices]

            # Get metadata for segments
            cluster_metadata = [self._search_index.metadata[i] for i in cluster_indices]

            # Compute scoring components
            breadth = self._compute_breadth(cluster_metadata)
            intensity = self._compute_intensity(cluster_metadata)
            coherence = self._compute_coherence(cluster_indices)
            recency = self._compute_recency(cluster_metadata)

            # Combined score
            score = (
                self.scoring_weights['breadth'] * breadth +
                self.scoring_weights['intensity'] * intensity +
                self.scoring_weights['coherence'] * coherence +
                self.scoring_weights['recency'] * recency
            )

            # Find representative segments (closest to centroid)
            representatives = self._find_representatives(cluster_indices, n=5)
            representative_segment_ids = [self._search_index.segment_ids[i] for i in representatives]

            # Extract channel and date info
            channels = list(set(
                m.get('channel_name') or m.get('channel_url', 'unknown')
                for m in cluster_metadata
            ))

            dates = [
                datetime.fromisoformat(m['publish_date'])
                for m in cluster_metadata
                if m.get('publish_date')
            ]
            date_range = (min(dates), max(dates)) if dates else (None, None)

            metrics.append(ClusterMetrics(
                cluster_id=int(cluster_id),
                size=len(segment_ids),
                segment_ids=segment_ids,
                breadth=breadth,
                intensity=intensity,
                coherence=coherence,
                recency=recency,
                score=score,
                representative_segment_ids=representative_segment_ids,
                channels=channels,
                date_range=date_range
            ))

        return metrics

    def _compute_breadth(self, metadata: List[Dict]) -> float:
        """
        Breadth: Unique channels discussing this topic (0-1 normalized)

        More channels = more significant topic
        """
        unique_channels = set(
            m.get('channel_name') or m.get('channel_url', 'unknown')
            for m in metadata
        )

        # Normalize by sqrt to avoid over-penalizing small clusters
        # Max expected: ~20 channels for 7d window
        max_channels = 20
        return min(1.0, len(unique_channels) / max_channels)

    def _compute_intensity(self, metadata: List[Dict]) -> float:
        """
        Intensity: Segments per day (0-1 normalized)

        More concentrated discussion = hotter topic
        """
        # Count segments per day
        dates = [
            datetime.fromisoformat(m['publish_date']).date()
            for m in metadata
            if m.get('publish_date')
        ]

        if not dates:
            return 0.0

        # Segments per day (average)
        date_range_days = (max(dates) - min(dates)).days + 1
        segments_per_day = len(metadata) / date_range_days

        # Normalize: expect max ~50 segments/day for hot topic in 7d window
        max_segments_per_day = 50
        return min(1.0, segments_per_day / max_segments_per_day)

    def _compute_coherence(self, cluster_indices: np.ndarray) -> float:
        """
        Coherence: Cluster tightness via HDBSCAN membership probabilities (0-1)

        Tighter cluster = better-defined topic
        """
        # Use HDBSCAN's membership probabilities
        cluster_probs = self._cluster_probabilities[cluster_indices]

        # Average probability (higher = more cohesive)
        return float(np.mean(cluster_probs))

    def _compute_recency(self, metadata: List[Dict]) -> float:
        """
        Recency: Time-weighted score favoring recent content (0-1)

        Uses exponential decay: recent segments weighted higher
        """
        dates = [
            datetime.fromisoformat(m['publish_date'])
            for m in metadata
            if m.get('publish_date')
        ]

        if not dates:
            return 0.0

        # Calculate days ago for each segment
        from datetime import timezone
        now = datetime.now(timezone.utc)

        # Make dates timezone-aware if they aren't already
        aware_dates = []
        for d in dates:
            if d.tzinfo is None:
                # Assume UTC for naive datetimes
                d = d.replace(tzinfo=timezone.utc)
            aware_dates.append(d)

        days_ago = [(now - d).days for d in aware_dates]

        # Exponential decay: weight = exp(-days_ago / half_life)
        # Half-life = time_window_days / 2 (middle of window gets 0.5 weight)
        half_life = self.time_window_days / 2
        weights = [np.exp(-d / half_life) for d in days_ago]

        # Average weight (normalized)
        return float(np.mean(weights))

    def _find_representatives(self, cluster_indices: np.ndarray, n: int = 5) -> List[int]:
        """
        Find representative segments closest to cluster centroid

        Args:
            cluster_indices: Indices of segments in cluster
            n: Number of representatives to return

        Returns:
            List of indices (into search_index arrays)
        """
        # Compute centroid in reduced space
        cluster_embeddings = self._reduced_embeddings[cluster_indices]
        centroid = np.mean(cluster_embeddings, axis=0)

        # Find closest segments to centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_indices = np.argsort(distances)[:n]

        # Return global indices (not cluster-local)
        return [cluster_indices[i] for i in closest_indices]

    def get_cluster_texts(self, cluster_id: int, max_samples: int = 10) -> List[Dict[str, Any]]:
        """
        Get text samples from a cluster for labeling/analysis

        Args:
            cluster_id: Cluster ID
            max_samples: Maximum number of samples to return

        Returns:
            List of dicts with {segment_id, text, channel, date}
        """
        if self._clusters is None:
            raise RuntimeError("Must run discover_topics() first")

        # Find cluster
        cluster = next((c for c in self._clusters if c.cluster_id == cluster_id), None)
        if cluster is None:
            raise ValueError(f"Cluster {cluster_id} not found")

        # Sample from representative segments first, then random
        sample_ids = cluster.representative_segment_ids[:max_samples]
        if len(sample_ids) < max_samples:
            # Add random samples
            remaining = max_samples - len(sample_ids)
            random_ids = np.random.choice(
                cluster.segment_ids,
                size=min(remaining, len(cluster.segment_ids)),
                replace=False
            ).tolist()
            sample_ids.extend([sid for sid in random_ids if sid not in sample_ids])

        # Get metadata
        samples = []
        for seg_id in sample_ids:
            # Find in search index
            try:
                idx = self._search_index.segment_ids.index(seg_id)
                meta = self._search_index.metadata[idx]

                samples.append({
                    'segment_id': seg_id,
                    'text': meta['text'],
                    'channel': meta.get('channel_name') or meta.get('channel_url'),
                    'date': meta.get('publish_date'),
                    'title': meta.get('title')
                })
            except ValueError:
                logger.warning(f"Segment {seg_id} not found in search index")
                continue

        return samples

    def discover_topics_for_group(
        self,
        group_segment_ids: List[int],
        group_id: str,
        force_rebuild: bool = False,
        sample_weights: Optional[np.ndarray] = None
    ) -> List[ClusterMetrics]:
        """
        Discover topics within a specific group of segments.

        This method scopes clustering to a subset of segments (a "group")
        rather than the full time window. Useful for hierarchical summarization
        where we want themes specific to ideological communities, languages, etc.

        Args:
            group_segment_ids: List of segment IDs belonging to this group
            group_id: Group identifier for logging
            force_rebuild: Force rebuild of search index
            sample_weights: Optional per-sample weights (must match length of group_segment_ids)

        Returns:
            List of ClusterMetrics for this group, sorted by score
        """
        logger.info(f"Starting group-scoped topic discovery for '{group_id}' ({len(group_segment_ids)} segments)")

        # Step 1: Get cached search index
        logger.info("Fetching search index...")
        self._search_index = self.search_service.get_index(
            self.time_window_days,
            force_rebuild=force_rebuild
        )

        if self._search_index.size() == 0:
            logger.warning("Empty search index, no topics to discover")
            return []

        # Step 2: Filter index to group segments only
        logger.info(f"Filtering index to group '{group_id}' segments...")
        group_segment_set = set(group_segment_ids)

        # Find indices in search index that belong to this group
        group_indices = []
        for idx, seg_id in enumerate(self._search_index.segment_ids):
            if seg_id in group_segment_set:
                group_indices.append(idx)

        if not group_indices:
            logger.warning(f"No segments from group '{group_id}' found in search index")
            return []

        logger.info(f"Found {len(group_indices)} group segments in index")

        # Step 3: Extract embeddings for group
        logger.info("Extracting group embeddings from FAISS index...")
        full_embeddings = self._extract_embeddings_from_faiss()
        self._embeddings = full_embeddings[group_indices]
        logger.info(f"Extracted group embeddings: {self._embeddings.shape}")

        # Create temporary search index view for this group
        self._group_indices = group_indices  # Store for later mapping

        # Step 4: Dimensionality reduction with UMAP
        logger.info(f"Running UMAP: {self._embeddings.shape[1]}d → {self.umap_n_components}d")
        self._reduced_embeddings = self._reduce_dimensions()
        logger.info(f"UMAP complete: {self._reduced_embeddings.shape}")

        # Step 5: Clustering with HDBSCAN
        logger.info("Running HDBSCAN clustering...")
        self._cluster_labels = self._cluster(sample_weights=sample_weights)

        unique_clusters = set(self._cluster_labels) - {-1}  # Exclude noise
        noise_count = np.sum(self._cluster_labels == -1)
        logger.info(
            f"HDBSCAN complete: {len(unique_clusters)} clusters, "
            f"{noise_count} noise points ({100*noise_count/len(self._cluster_labels):.1f}%)"
        )

        # Step 6: Compute metrics for group-scoped clusters
        logger.info("Computing cluster metrics and scores...")
        self._clusters = self._compute_group_cluster_metrics(group_indices)

        # Sort by score
        self._clusters.sort(key=lambda c: c.score, reverse=True)

        logger.info(f"✓ Group topic discovery complete for '{group_id}': {len(self._clusters)} topics")
        if self._clusters:
            logger.info(f"  Top topic: score={self._clusters[0].score:.3f}, size={self._clusters[0].size}")

        return self._clusters

    def _compute_group_cluster_metrics(self, group_indices: List[int]) -> List[ClusterMetrics]:
        """
        Compute metrics for clusters within a group.

        Similar to _compute_cluster_metrics but uses group-filtered indices.

        Args:
            group_indices: Indices in search_index that belong to this group

        Returns:
            List of ClusterMetrics
        """
        metrics = []
        unique_clusters = set(self._cluster_labels) - {-1}  # Exclude noise

        for cluster_id in unique_clusters:
            # Get segments in this cluster (using group-local indices)
            mask = self._cluster_labels == cluster_id
            cluster_local_indices = np.where(mask)[0]

            # Map back to global search index indices
            cluster_global_indices = [group_indices[i] for i in cluster_local_indices]
            segment_ids = [self._search_index.segment_ids[i] for i in cluster_global_indices]

            # Get metadata for segments
            cluster_metadata = [self._search_index.metadata[i] for i in cluster_global_indices]

            # Compute scoring components
            breadth = self._compute_breadth(cluster_metadata)
            intensity = self._compute_intensity(cluster_metadata)
            coherence = self._compute_coherence(cluster_local_indices)
            recency = self._compute_recency(cluster_metadata)

            # Combined score
            score = (
                self.scoring_weights['breadth'] * breadth +
                self.scoring_weights['intensity'] * intensity +
                self.scoring_weights['coherence'] * coherence +
                self.scoring_weights['recency'] * recency
            )

            # Find representative segments (in group-local space)
            representatives_local = self._find_representatives(cluster_local_indices, n=5)
            # Map to global indices
            representative_global_indices = [group_indices[i] for i in representatives_local]
            representative_segment_ids = [self._search_index.segment_ids[i] for i in representative_global_indices]

            # Extract channel and date info
            channels = list(set(
                m.get('channel_name') or m.get('channel_url', 'unknown')
                for m in cluster_metadata
            ))

            dates = [
                datetime.fromisoformat(m['publish_date'])
                for m in cluster_metadata
                if m.get('publish_date')
            ]
            date_range = (min(dates), max(dates)) if dates else (None, None)

            metrics.append(ClusterMetrics(
                cluster_id=int(cluster_id),
                size=len(segment_ids),
                segment_ids=segment_ids,
                breadth=breadth,
                intensity=intensity,
                coherence=coherence,
                recency=recency,
                score=score,
                representative_segment_ids=representative_segment_ids,
                channels=channels,
                date_range=date_range
            ))

        return metrics
