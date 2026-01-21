#!/usr/bin/env python3
"""
Generate Embedding Clusters with BERTopic
==========================================

Pre-compute topic clusters using BERTopic (UMAP + HDBSCAN + c-TF-IDF).

CLUSTERING APPROACH:
    BERTopic provides state-of-the-art topic modeling:
    - UMAP: Dimensionality reduction (preserves semantic structure)
    - HDBSCAN: Density-based clustering (finds natural topic boundaries)
    - c-TF-IDF: Extracts topic keywords automatically

    Benefits over manual HDBSCAN:
    - Adaptive parameters based on dataset size and time window
    - Hierarchical topic reduction to target number of themes
    - Automatic topic naming via c-TF-IDF keywords
    - Better handling of diverse content (no mega-clusters)

UNIFIED CACHE ARCHITECTURE:
    All cache tables contain BOTH embedding models:
    - embedding: main model (qwen3:0.6b, 1024-dim)
    - embedding_alt: alt model (qwen3:4b, 2000-dim)

Usage:
    python generate_embedding_clusters.py --time-window 30d --model main
    python generate_embedding_clusters.py --all  # Run all 4 combinations

Options:
    --time-window   Time window: 30d or 7d
    --model         Model: main (0.6b) or alt (4b)
    --all           Run all combinations
    --max-themes    Maximum themes to keep (default: 20)
    --min-size      Min cluster size (default: 200, adaptive)
    --sample        Sample size (default: all)
    --dry-run       Preview without storing
    --projects      Filter by projects
    --languages     Filter by languages

Adaptive Parameters:
    7d: min_size ~186 (0.2%), target 15 topics (breaking news)
    30d: min_size ~360 (0.1%), target 20 topics (broader themes)

Performance:
    30d (360k segments): ~5-10 minutes
    7d (93k segments): ~1-3 minutes
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder
from hdbscan import HDBSCAN
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection - use centralized config
def _get_db_config():
    """Get database config from environment variables."""
    from dotenv import load_dotenv
    from src.utils.paths import get_env_path
    env_path = get_env_path()
    if env_path.exists():
        load_dotenv(env_path)

    password = os.getenv('POSTGRES_PASSWORD')
    if not password:
        raise ValueError("POSTGRES_PASSWORD environment variable is required")

    return {
        'host': os.getenv('POSTGRES_HOST', '10.0.0.4'),
        'database': os.getenv('POSTGRES_DB', 'av_content'),
        'user': os.getenv('POSTGRES_USER', 'signal4'),
        'password': password,
    }

DB_CONFIG = _get_db_config()


class PrecomputedEmbeddings(BaseEmbedder):
    """Wrapper for pre-computed embeddings that BERTopic can use."""

    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def embed(self, documents, verbose=False):
        """Return pre-computed embeddings."""
        return self.embeddings


def get_connection():
    """Get database connection."""
    return psycopg2.connect(**DB_CONFIG)


def fetch_embeddings(conn, time_window, model_version, projects_filter=None, languages_filter=None, sample_size=None):
    """
    Fetch embeddings from unified cache table.

    Cache tables now contain BOTH embedding models:
    - embedding column: main model (qwen3:0.6b, 1024-dim)
    - embedding_alt column: alt model (qwen3:4b, 2000-dim)

    Returns:
        List of dicts with: id, content_id, content_id_string,
                           embedding, projects, main_language, publish_date
    """
    # Unified cache table (no more separate _alt tables)
    table = f"embedding_cache_{time_window}"

    # Select correct embedding column based on model
    embedding_col = 'embedding' if model_version == 'main' else 'embedding_alt'

    # Build query
    where_clauses = [f"{embedding_col} IS NOT NULL"]
    params = []

    if projects_filter:
        where_clauses.append("projects && %s::varchar[]")
        params.append(projects_filter)

    if languages_filter:
        where_clauses.append("main_language = ANY(%s)")
        params.append(languages_filter)

    where_sql = " AND ".join(where_clauses)

    # Add sampling if requested
    sample_sql = ""
    if sample_size:
        sample_sql = f"LIMIT {sample_size}"

    sql = f"""
        SELECT
            id,
            content_id,
            content_id_string,
            segment_index,
            {embedding_col} as embedding,
            text,
            projects,
            main_language,
            publish_date
        FROM {table}
        WHERE {where_sql}
        {sample_sql}
    """

    logger.info(f"Fetching {model_version} embeddings from {table}...")
    logger.debug(f"SQL: {sql}")
    logger.debug(f"Params: {params}")

    start = time.time()

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(sql, params)
        results = cur.fetchall()

    duration = time.time() - start
    logger.info(f"Fetched {len(results):,} segments in {duration:.1f}s")

    # Convert embedding strings to numpy arrays
    for row in results:
        if isinstance(row['embedding'], str):
            # Parse JSON array string to list, then to numpy array
            row['embedding'] = np.array(json.loads(row['embedding']), dtype=np.float32)
        elif row['embedding'] is not None:
            # Already a list or array
            row['embedding'] = np.array(row['embedding'], dtype=np.float32)

    return results


def cluster_embeddings(embeddings, documents, time_window, min_cluster_size=200, max_themes=20):
    """
    Cluster embeddings using BERTopic (UMAP + HDBSCAN + c-TF-IDF).

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        documents: list of document texts for c-TF-IDF
        time_window: '7d' or '30d' for adaptive parameters
        min_cluster_size: Minimum cluster size for HDBSCAN
        max_themes: Maximum number of themes to return

    Returns:
        topics: numpy array of topic IDs (-1 = outlier)
        topic_model: BERTopic model with metadata
        cluster_info: dict with clustering metrics
    """
    n_samples = len(embeddings)
    logger.info(f"Clustering {n_samples:,} embeddings with BERTopic...")

    start = time.time()

    # Adaptive parameters based on dataset size and time window
    if time_window == '7d':
        # 7d: More granular, focused topics
        adaptive_min_size = max(100, min(500, int(n_samples * 0.002)))
        target_topics = 15
    else:  # 30d
        # 30d: Broader themes
        adaptive_min_size = max(200, min(1000, int(n_samples * 0.001)))
        target_topics = 20

    # Override with user-provided value if specified
    final_min_size = min_cluster_size if min_cluster_size != 200 else adaptive_min_size
    final_max_themes = max_themes

    logger.info(f"Using min_cluster_size={final_min_size}, target_topics={target_topics}, max_themes={final_max_themes}")

    # Configure UMAP with higher n_neighbors to preserve global structure
    # This helps prevent language-based clustering by looking at more neighbors
    umap_model = UMAP(
        n_neighbors=min(50, n_samples - 1),  # Increased from 15 to capture more global structure
        n_components=min(10, n_samples - 1),  # Increased from 5 for richer representation
        min_dist=0.0,  # Allow tight clusters
        metric='cosine',
        random_state=42
    )

    # Configure HDBSCAN with lower min_samples to reduce noise
    hdbscan_model = HDBSCAN(
        min_cluster_size=final_min_size,
        min_samples=max(10, final_min_size // 5),  # Lower to reduce outliers
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True
    )

    # Configure vectorizer for c-TF-IDF with safe parameters for small datasets
    # No max_df to avoid errors with small language groups
    vectorizer_model = CountVectorizer(
        stop_words="english",  # Will catch some common words
        min_df=1,  # Keep all words (c-TF-IDF will handle relevance)
        ngram_range=(1, 2)  # Unigrams and bigrams
    )

    # Wrap embeddings for BERTopic (needed for outlier reduction)
    embedding_model = PrecomputedEmbeddings(embeddings)

    # Initialize BERTopic with pre-computed embeddings
    topic_model = BERTopic(
        embedding_model=embedding_model,  # Wrapped pre-computed embeddings
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        nr_topics=target_topics,  # Automatically reduce to this many topics
        calculate_probabilities=False,  # Skip for performance
        verbose=True
    )

    # Fit transform with pre-computed embeddings
    logger.info("Running BERTopic fit_transform...")
    topics, probs = topic_model.fit_transform(documents, embeddings)

    total_duration = time.time() - start
    logger.info(f"BERTopic clustering complete in {total_duration:.1f}s")

    # Analyze initial results
    unique_topics = [t for t in np.unique(topics) if t >= 0]
    topic_sizes = {int(t): int(np.sum(topics == t)) for t in unique_topics}
    noise_count = int(np.sum(topics == -1))

    logger.info(f"Initial: {len(unique_topics)} topics (+ {noise_count} outliers)")

    # Reduce outliers by reassigning to semantic topics using embedding similarity
    # Only run if we have topics AND outliers
    if len(unique_topics) > 0 and noise_count > 0:
        logger.info("Running outlier reduction with embeddings strategy...")
        new_topics = topic_model.reduce_outliers(documents, topics, strategy="embeddings")

        # Update topic assignments
        topics = new_topics

        # Recalculate results after outlier reduction
        unique_topics = [t for t in np.unique(topics) if t >= 0]
        topic_sizes = {int(t): int(np.sum(topics == t)) for t in unique_topics}
        noise_count = int(np.sum(topics == -1))

        logger.info(f"After outlier reduction: {len(unique_topics)} topics (+ {noise_count} outliers)")
    elif len(unique_topics) == 0:
        logger.info("Skipping outlier reduction - no topics found")
    else:
        logger.info("Skipping outlier reduction - no outliers")

    # Show top topics
    sorted_topics = sorted(topic_sizes.items(), key=lambda x: x[1], reverse=True)
    logger.info(f"  Top {min(20, len(sorted_topics))} topics by size:")
    for topic_id, size in sorted_topics[:20]:
        pct = 100 * size / n_samples
        # Get top words for this topic
        topic_words = topic_model.get_topic(topic_id)
        top_words = ", ".join([word for word, _ in topic_words[:5]]) if topic_words else "N/A"
        logger.info(f"    Topic {topic_id:3d}: {size:6,} segments ({pct:4.1f}%) - {top_words}")

    # Limit to max_themes if needed
    if len(sorted_topics) > final_max_themes:
        logger.info(f"Reducing from {len(sorted_topics)} to {final_max_themes} topics...")

        # Mark topics beyond max_themes as outliers
        keep_topics = {t for t, _ in sorted_topics[:final_max_themes]}
        for t, _ in sorted_topics[final_max_themes:]:
            topics[topics == t] = -1

        # Update statistics
        sorted_topics = sorted_topics[:final_max_themes]
        noise_count = int(np.sum(topics == -1))

    cluster_info = {
        'num_clusters': len(sorted_topics),
        'cluster_sizes': dict(sorted_topics),
        'noise_count': noise_count,
        'total_duration': total_duration,
        'parameters': {
            'min_cluster_size': final_min_size,
            'max_themes': final_max_themes,
            'target_topics': target_topics,
            'time_window': time_window
        }
    }

    logger.info(f"Final result: {len(sorted_topics)} topics, {noise_count:,} outliers")

    return topics, topic_model, cluster_info


def cluster_embeddings_language_stratified(embeddings, documents, segments, time_window, min_cluster_size=200, max_themes=20, alignment_threshold=0.75):
    """
    Language-stratified clustering: cluster each language separately, then align topics across languages.

    This approach:
    1. Clusters each language independently (avoids generic language mega-clusters)
    2. Calculates topic centroids for each language
    3. Aligns similar topics across languages using centroid cosine similarity
    4. Assigns cross_lingual_topic_id to aligned topics

    Args:
        embeddings: numpy array of shape (n_samples, n_features)
        documents: list of document texts for c-TF-IDF
        segments: list of segment dicts with 'main_language' field
        time_window: '7d' or '30d' for adaptive parameters
        min_cluster_size: Minimum cluster size for HDBSCAN
        max_themes: Maximum number of themes per language
        alignment_threshold: Cosine similarity threshold for cross-lingual alignment (default 0.75)

    Returns:
        topics: numpy array of language-specific topic IDs (-1 = outlier)
        cross_lingual_topics: numpy array of cross-lingual topic IDs (-1 = no alignment)
        topic_models: dict mapping language → BERTopic model
        cluster_info: dict with clustering metrics including alignment info
    """
    logger.info(f"Running language-stratified clustering...")
    start = time.time()

    # Group segments by language
    language_groups = {}
    for i, seg in enumerate(segments):
        lang = seg.get('main_language', 'unknown')
        if lang not in language_groups:
            language_groups[lang] = []
        language_groups[lang].append(i)

    # Filter out languages with too few samples
    min_samples_per_lang = max(100, min_cluster_size)
    language_groups = {lang: indices for lang, indices in language_groups.items()
                       if len(indices) >= min_samples_per_lang}

    logger.info(f"Found {len(language_groups)} languages with sufficient samples:")
    for lang, indices in sorted(language_groups.items(), key=lambda x: len(x[1]), reverse=True):
        logger.info(f"  {lang}: {len(indices):,} segments")

    # Cluster each language separately
    all_topics = np.full(len(embeddings), -1, dtype=int)
    topic_models = {}
    language_centroids = {}  # lang → {topic_id: centroid}
    language_topic_info = {}  # lang → {topic_id: {size, keywords}}
    topic_offset = 0

    for lang, indices in sorted(language_groups.items()):
        logger.info(f"\nClustering {lang} ({len(indices):,} segments)...")

        lang_embeddings = embeddings[indices]
        lang_documents = [documents[i] for i in indices]

        # Adaptive min_cluster_size based on language sample size
        # Aim for ~0.5-1% of data in each cluster, with reasonable bounds
        lang_min_size = max(50, min(500, int(len(indices) * 0.005)))
        logger.info(f"  Using language-specific min_cluster_size={lang_min_size}")

        # Cluster this language
        lang_topics, lang_model, lang_cluster_info = cluster_embeddings(
            lang_embeddings, lang_documents, time_window, lang_min_size, max_themes
        )

        # Remap topic IDs to be globally unique (offset by previous languages)
        lang_topics_remapped = lang_topics.copy()
        lang_topics_remapped[lang_topics >= 0] += topic_offset

        # Store in global topics array
        for i, global_idx in enumerate(indices):
            all_topics[global_idx] = lang_topics_remapped[i]

        topic_models[lang] = lang_model

        # Calculate centroids for this language's topics
        lang_centroids = {}
        lang_topic_data = {}
        unique_topics = [t for t in np.unique(lang_topics) if t >= 0]

        for topic_id in unique_topics:
            mask = lang_topics == topic_id
            topic_embeddings = lang_embeddings[mask]
            centroid = np.mean(topic_embeddings, axis=0)

            global_topic_id = topic_id + topic_offset
            lang_centroids[global_topic_id] = centroid

            # Get keywords
            topic_words = lang_model.get_topic(topic_id)
            keywords = ", ".join([word for word, _ in topic_words[:5]]) if topic_words else "N/A"

            lang_topic_data[global_topic_id] = {
                'size': int(np.sum(mask)),
                'keywords': keywords,
                'local_id': topic_id
            }

        language_centroids[lang] = lang_centroids
        language_topic_info[lang] = lang_topic_data

        # Update offset for next language
        if unique_topics:
            topic_offset = max(lang_topics_remapped) + 1

    logger.info(f"\nLanguage-specific clustering complete. Total topics: {topic_offset}")

    # Cross-lingual topic alignment
    logger.info(f"Aligning topics across languages (threshold: {alignment_threshold})...")

    cross_lingual_topics = np.full(len(embeddings), -1, dtype=int)
    cross_lingual_id = 0
    aligned_topics = {}  # global_topic_id → cross_lingual_id

    # Build list of all topic centroids with metadata
    all_centroids = []
    for lang, centroids in language_centroids.items():
        for topic_id, centroid in centroids.items():
            all_centroids.append({
                'language': lang,
                'topic_id': topic_id,
                'centroid': centroid,
                'size': language_topic_info[lang][topic_id]['size'],
                'keywords': language_topic_info[lang][topic_id]['keywords']
            })

    # Sort by size (largest first) to prioritize major topics
    all_centroids.sort(key=lambda x: x['size'], reverse=True)

    # Greedy alignment: process topics by size, align similar ones
    for topic_data in all_centroids:
        topic_id = topic_data['topic_id']

        if topic_id in aligned_topics:
            continue  # Already aligned

        # Find all similar topics across languages
        similar_topics = [topic_id]
        centroid = topic_data['centroid']

        for other_data in all_centroids:
            other_id = other_data['topic_id']
            if other_id in aligned_topics or other_id == topic_id:
                continue

            # Calculate cosine similarity
            other_centroid = other_data['centroid']
            similarity = np.dot(centroid, other_centroid) / (
                np.linalg.norm(centroid) * np.linalg.norm(other_centroid)
            )

            if similarity >= alignment_threshold:
                similar_topics.append(other_id)

        # Assign cross-lingual ID to all similar topics
        for t_id in similar_topics:
            aligned_topics[t_id] = cross_lingual_id

        cross_lingual_id += 1

    # Map segments to cross-lingual topics
    for i, topic_id in enumerate(all_topics):
        if topic_id >= 0 and topic_id in aligned_topics:
            cross_lingual_topics[i] = aligned_topics[topic_id]

    # Report alignment results
    logger.info(f"Aligned {topic_offset} language-specific topics into {cross_lingual_id} cross-lingual topics")

    # Analyze alignment quality
    mono_lingual = 0
    multi_lingual = 0
    for cl_id in range(cross_lingual_id):
        # Find which languages are in this cross-lingual topic
        languages = set()
        for topic_id, cl_topic_id in aligned_topics.items():
            if cl_topic_id == cl_id:
                for lang, centroids in language_centroids.items():
                    if topic_id in centroids:
                        languages.add(lang)
                        break

        if len(languages) == 1:
            mono_lingual += 1
        else:
            multi_lingual += 1

    logger.info(f"  Cross-lingual topics: {multi_lingual}")
    logger.info(f"  Language-specific topics: {mono_lingual}")

    total_duration = time.time() - start

    cluster_info = {
        'num_clusters': topic_offset,
        'num_cross_lingual_topics': cross_lingual_id,
        'num_languages': len(language_groups),
        'multi_lingual_topics': multi_lingual,
        'mono_lingual_topics': mono_lingual,
        'alignment_threshold': alignment_threshold,
        'total_duration': total_duration,
        'language_groups': {lang: len(indices) for lang, indices in language_groups.items()},
        'aligned_topics': aligned_topics,
        'language_topic_info': language_topic_info
    }

    return all_topics, cross_lingual_topics, topic_models, cluster_info


def calculate_cluster_metadata(embeddings, topics, segments, topic_model):
    """
    Calculate topic centroids, distances, and extract keywords from BERTopic.

    Returns:
        topic_centroids: dict mapping topic_id → centroid vector
        segment_distances: dict mapping segment_id → distance to centroid
        representative_segments: dict mapping topic_id → list of representative segment IDs
        topic_keywords: dict mapping topic_id → list of (word, score) tuples
    """
    topic_centroids = {}
    segment_distances = {}
    representative_segments = {}
    topic_keywords = {}

    unique_topics = [t for t in np.unique(topics) if t >= 0]

    for topic_id in unique_topics:
        mask = topics == topic_id
        topic_embeddings = embeddings[mask]
        topic_segment_ids = [segments[i]['id'] for i, m in enumerate(mask) if m]

        # Calculate centroid
        centroid = topic_embeddings.mean(axis=0)
        topic_centroids[topic_id] = centroid

        # Calculate distances to centroid
        distances = np.linalg.norm(topic_embeddings - centroid, axis=1)

        for i, (seg_id, dist) in enumerate(zip(topic_segment_ids, distances)):
            segment_distances[seg_id] = float(dist)

        # Find representative segments (closest to centroid)
        sorted_indices = np.argsort(distances)
        representative_segments[topic_id] = [topic_segment_ids[i] for i in sorted_indices[:10]]

        # Extract topic keywords from BERTopic
        topic_words = topic_model.get_topic(topic_id)
        if topic_words:
            topic_keywords[topic_id] = [(word, float(score)) for word, score in topic_words[:10]]
        else:
            topic_keywords[topic_id] = []

    return topic_centroids, segment_distances, representative_segments, topic_keywords


def store_clusters(conn, time_window, model_version, segments, cluster_labels, cluster_info, cluster_centroids, segment_distances, representative_segments, topic_keywords, cross_lingual_topics=None, dry_run=False):
    """
    Store clustering results in database.

    Args:
        cross_lingual_topics: Optional array of cross-lingual topic IDs (for language-stratified clustering)
    """
    if dry_run:
        logger.info("[DRY RUN] Would store clustering results")
        return

    clustered_at = datetime.now()

    logger.info("Storing clustering results...")

    # Prepare batch insert data
    rows = []
    for i, segment in enumerate(segments):
        # Convert numpy types to native Python types
        cluster_id = int(cluster_labels[i])

        # Skip if cluster was filtered out
        if cluster_id >= 0 and cluster_id not in cluster_centroids:
            cluster_id = -1

        # Convert numpy arrays to lists for PostgreSQL
        centroid = cluster_centroids.get(cluster_id)
        if centroid is not None and isinstance(centroid, np.ndarray):
            centroid = centroid.tolist()

        # Get distance, converting numpy types to Python float
        distance = segment_distances.get(segment['id'])
        if distance is not None:
            distance = float(distance)

        # Get cross-lingual topic ID if available
        cross_lingual_id = None
        if cross_lingual_topics is not None:
            cross_lingual_id = int(cross_lingual_topics[i]) if cross_lingual_topics[i] >= 0 else None

        # Get topic name from keywords
        keywords = topic_keywords.get(cluster_id, [])
        if keywords and cluster_id >= 0:
            top_words = [word for word, _ in keywords[:3]]
            cluster_name = ", ".join(top_words)
        elif cluster_id >= 0:
            cluster_name = f"Topic {cluster_id}"
        else:
            cluster_name = "Outlier"

        row = {
            'segment_id': int(segment['id']),
            'content_id': int(segment['content_id']),
            'content_id_string': segment['content_id_string'],
            'time_window': time_window,
            'model_version': model_version,
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'cluster_size': int(cluster_info.get('cluster_sizes', {}).get(cluster_id, 0)) if cluster_id >= 0 else 0,
            'cluster_centroid': centroid,
            'distance_to_centroid': distance,
            'is_representative': segment['id'] in representative_segments.get(cluster_id, []) if cluster_id >= 0 else False,
            'cross_lingual_topic_id': cross_lingual_id,
            'projects': segment['projects'],
            'main_language': segment['main_language'],
            'publish_date': segment['publish_date'],
            'clustering_method': 'bertopic_language_stratified' if cross_lingual_topics is not None else 'bertopic',
            'clustering_params': json.dumps(cluster_info.get('parameters', {})),
            'clustered_at': clustered_at
        }
        rows.append(row)

    # Batch insert
    start = time.time()

    with conn.cursor() as cur:
        execute_batch(cur, """
            INSERT INTO embedding_clusters (
                segment_id, content_id, content_id_string,
                time_window, model_version, cluster_id, cluster_name, cluster_size,
                cluster_centroid, distance_to_centroid, is_representative,
                cross_lingual_topic_id,
                projects, main_language, publish_date,
                clustering_method, clustering_params, clustered_at
            ) VALUES (
                %(segment_id)s, %(content_id)s, %(content_id_string)s,
                %(time_window)s, %(model_version)s, %(cluster_id)s, %(cluster_name)s, %(cluster_size)s,
                %(cluster_centroid)s, %(distance_to_centroid)s, %(is_representative)s,
                %(cross_lingual_topic_id)s,
                %(projects)s, %(main_language)s, %(publish_date)s,
                %(clustering_method)s, %(clustering_params)s, %(clustered_at)s
            )
        """, rows, page_size=1000)

        conn.commit()

    duration = time.time() - start
    logger.info(f"Stored {len(rows):,} cluster assignments in {duration:.1f}s")

    # Store cluster metadata
    meta_rows = []
    for cluster_id, centroid in cluster_centroids.items():
        # Convert numpy array to list for PostgreSQL
        if isinstance(centroid, np.ndarray):
            centroid = centroid.tolist()

        # Get topic name from keywords
        keywords = topic_keywords.get(cluster_id, [])
        if keywords:
            top_words = [word for word, _ in keywords[:3]]
            cluster_name = ", ".join(top_words)
        else:
            cluster_name = f"Topic {cluster_id}"

        meta_rows.append({
            'time_window': time_window,
            'model_version': model_version,
            'clustered_at': clustered_at,
            'cluster_id': int(cluster_id),
            'cluster_name': cluster_name,
            'cluster_size': int(cluster_info['cluster_sizes'][cluster_id]),
            'cluster_centroid': centroid,
            'representative_segment_ids': [int(seg_id) for seg_id in representative_segments[cluster_id]],
            'silhouette_score': None,  # BERTopic doesn't compute this by default
            'clustering_method': 'bertopic',
            'clustering_params': json.dumps({
                **cluster_info['parameters'],
                'topic_keywords': keywords  # Add keywords to metadata
            })
        })

    with conn.cursor() as cur:
        execute_batch(cur, """
            INSERT INTO cluster_metadata (
                time_window, model_version, clustered_at, cluster_id, cluster_name,
                cluster_size, cluster_centroid, representative_segment_ids,
                silhouette_score, clustering_method, clustering_params
            ) VALUES (
                %(time_window)s, %(model_version)s, %(clustered_at)s, %(cluster_id)s, %(cluster_name)s,
                %(cluster_size)s, %(cluster_centroid)s, %(representative_segment_ids)s,
                %(silhouette_score)s, %(clustering_method)s, %(clustering_params)s
            )
        """, meta_rows)

        conn.commit()

    logger.info(f"Stored {len(meta_rows)} cluster metadata records")


def main():
    parser = argparse.ArgumentParser(description='Generate embedding clusters')
    parser.add_argument('--time-window', choices=['30d', '7d'], help='Time window')
    parser.add_argument('--model', choices=['main', 'alt'], help='Model version')
    parser.add_argument('--all', action='store_true', help='Run all combinations (30d+7d × main+alt)')
    parser.add_argument('--max-themes', type=int, default=20, help='Maximum number of themes to keep')
    parser.add_argument('--min-size', type=int, default=200, help='Minimum cluster size (HDBSCAN parameter - higher = fewer, larger clusters)')
    parser.add_argument('--min-pct', type=float, default=0.5, help='Minimum cluster percentage (volume filter - lower = more themes kept)')
    parser.add_argument('--sample', type=int, help='Sample size for clustering')
    parser.add_argument('--projects', help='Filter by projects (comma-separated)')
    parser.add_argument('--languages', help='Filter by languages (comma-separated)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    args = parser.parse_args()

    # Validate arguments
    if args.all:
        if args.time_window or args.model:
            parser.error("--all cannot be combined with --time-window or --model")

        # Run all combinations
        combinations = [
            ('30d', 'main'),
            ('30d', 'alt'),
            ('7d', 'main'),
            ('7d', 'alt')
        ]

        logger.info("=" * 80)
        logger.info("GENERATE EMBEDDING CLUSTERS: ALL COMBINATIONS")
        logger.info("=" * 80)
        logger.info(f"Running {len(combinations)} clustering jobs...")
        logger.info("")

        all_results = []
        overall_start = time.time()

        for i, (time_window, model) in enumerate(combinations, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"JOB {i}/{len(combinations)}: {time_window} / {model}")
            logger.info(f"{'='*80}")

            try:
                result = run_clustering(
                    time_window=time_window,
                    model=model,
                    max_themes=args.max_themes,
                    min_size=args.min_size,
                    min_pct=args.min_pct,
                    sample_size=args.sample,
                    projects_filter=args.projects,
                    languages_filter=args.languages,
                    dry_run=args.dry_run
                )
                all_results.append((time_window, model, result))
            except Exception as e:
                logger.error(f"Failed to cluster {time_window}/{model}: {e}")
                all_results.append((time_window, model, None))

        # Final summary
        overall_duration = time.time() - overall_start
        logger.info("\n" + "=" * 80)
        logger.info("ALL JOBS COMPLETE")
        logger.info("=" * 80)

        for time_window, model, result in all_results:
            if result:
                logger.info(f"{time_window:5s} / {model:4s}: ✓ {result['num_clusters']} clusters, {result['total_segments']:,} segments")
            else:
                logger.info(f"{time_window:5s} / {model:4s}: ✗ FAILED")

        logger.info(f"\nTotal time: {overall_duration:.1f}s ({overall_duration/60:.1f}m)")
        logger.info("=" * 80)

    else:
        # Single run mode
        if not args.time_window or not args.model:
            parser.error("--time-window and --model are required unless using --all")

        result = run_clustering(
            time_window=args.time_window,
            model=args.model,
            max_themes=args.max_themes,
            min_size=args.min_size,
            min_pct=args.min_pct,
            sample_size=args.sample,
            projects_filter=args.projects,
            languages_filter=args.languages,
            dry_run=args.dry_run
        )


def run_clustering(time_window, model, max_themes, min_size, min_pct, sample_size, projects_filter, languages_filter, dry_run):
    """Run clustering for a single time window and model combination."""

    logger.info("=" * 80)
    logger.info(f"GENERATE EMBEDDING CLUSTERS: {time_window} / {model}")
    logger.info("=" * 80)

    projects_filter = projects_filter.split(',') if projects_filter else None
    languages_filter = languages_filter.split(',') if languages_filter else None

    conn = get_connection()

    try:
        total_start = time.time()

        # 1. Fetch embeddings
        segments = fetch_embeddings(
            conn,
            time_window,
            model,
            projects_filter,
            languages_filter,
            sample_size
        )

        if not segments:
            logger.error("No segments found!")
            return None

        # 2. Extract embeddings and texts as numpy array
        embeddings = np.vstack([s['embedding'] for s in segments])
        documents = [s.get('text', '') or '' for s in segments]  # Get text for c-TF-IDF
        logger.info(f"Embeddings shape: {embeddings.shape}")

        # 3. Cluster with language-stratified BERTopic
        topics, cross_lingual_topics, topic_models, cluster_info = cluster_embeddings_language_stratified(
            embeddings,
            documents,
            segments,
            time_window,
            min_cluster_size=min_size,
            max_themes=max_themes,
            alignment_threshold=0.75
        )

        # Use first language's model for metadata calculation (they all use same structure)
        topic_model = next(iter(topic_models.values()))

        # 4. Calculate cluster metadata
        logger.info("Calculating cluster metadata...")
        cluster_centroids, segment_distances, representative_segments, topic_keywords = calculate_cluster_metadata(
            embeddings,
            topics,
            segments,
            topic_model
        )

        # 5. Store results
        store_clusters(
            conn,
            time_window,
            model,
            segments,
            topics,
            cluster_info,
            cluster_centroids,
            segment_distances,
            representative_segments,
            topic_keywords,
            cross_lingual_topics=cross_lingual_topics,
            dry_run=dry_run
        )

        total_duration = time.time() - total_start

        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Segments:          {len(segments):,}")
        logger.info(f"Clusters:          {cluster_info['num_clusters']}")
        logger.info(f"Noise points:      {cluster_info['noise_count']:,}")
        logger.info(f"Silhouette score:  {cluster_info.get('silhouette_score', 'N/A')}")
        logger.info(f"Total time:        {total_duration:.1f}s ({total_duration/60:.1f}m)")
        logger.info("=" * 80)

        if not dry_run:
            logger.info("✓ Clusters stored successfully")

        # Return summary for --all mode
        return {
            'num_clusters': cluster_info['num_clusters'],
            'total_segments': len(segments),
            'noise_count': cluster_info['noise_count'],
            'duration': total_duration
        }

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return None
    finally:
        conn.close()


if __name__ == '__main__':
    main()
