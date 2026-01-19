#!/usr/bin/env python3
"""
Test script for segment reranking pipeline.

Tests the full flow:
1. Query expansion (without LLM - use static queries)
2. Semantic search via pgvector
3. Reranking with diversity constraints
4. Performance measurement

Usage:
    cd ~/signal4/core
    uv run python scripts/test_reranking.py
"""

import asyncio
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime, timezone, timedelta
import numpy as np


async def test_reranking():
    """Test the reranking pipeline end-to-end."""

    print("=" * 60)
    print("Segment Reranking Pipeline Test")
    print("=" * 60)

    from src.backend.app.services.pgvector_search_service import PgVectorSearchService

    # Import reranker directly to avoid pulling in umap via __init__.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "segment_reranker",
        Path(__file__).parent.parent / "src/backend/app/services/rag/segment_reranker.py"
    )
    segment_reranker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(segment_reranker)
    SegmentReranker = segment_reranker.SegmentReranker
    RerankerWeights = segment_reranker.RerankerWeights
    DiversityConstraints = segment_reranker.DiversityConstraints

    # Create a mock config for search service
    class MockConfig:
        project = "CPRMV"
        allowed_projects = ["CPRMV"]
        use_alt_embeddings = False

    config = MockConfig()
    dashboard_id = "test_reranking"

    # Initialize services
    search_service = PgVectorSearchService(dashboard_id, config)

    print("\n1. Fetching a real embedding from database to use as query...")
    t0 = time.perf_counter()

    # Get a real embedding from the cache table to use as query
    # This ensures we get actual matches
    import psycopg2
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(
        host='10.0.0.4',
        database='av_content',
        user='signal4',
        password='signal4'
    )
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT embedding FROM embedding_cache_30d
        WHERE embedding IS NOT NULL
        LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row or not row['embedding']:
        print("   ERROR: No embeddings found in cache table")
        return

    # Parse embedding from pgvector string format
    emb_str = row['embedding']
    if isinstance(emb_str, str):
        emb_str = emb_str.strip('[]')
        query_embedding = np.array([float(x) for x in emb_str.split(',')], dtype=np.float32)
    else:
        query_embedding = np.array(emb_str, dtype=np.float32)

    t1 = time.perf_counter()
    print(f"   Embedding ready (dim={len(query_embedding)}): {(t1-t0)*1000:.1f}ms")

    # Test parameters
    time_window_days = 30
    k = 500  # Retrieve many, then rerank
    threshold = 0.35  # Lower threshold to get more candidates

    print(f"\n2. Searching pgvector (k={k}, threshold={threshold}, window={time_window_days}d)...")
    t0 = time.perf_counter()

    results = search_service.search(
        query_embedding=query_embedding,
        time_window_days=time_window_days,
        k=k,
        threshold=threshold
    )

    t1 = time.perf_counter()
    search_time = (t1 - t0) * 1000
    print(f"   Retrieved {len(results)} segments in {search_time:.1f}ms")

    if not results:
        print("   No results found. Try adjusting threshold or time window.")
        return

    # Show sample of raw results
    print(f"\n   Sample raw results (top 5 by similarity):")
    for i, seg in enumerate(results[:5]):
        print(f"   [{i+1}] sim={seg['similarity']:.3f} | {seg['channel_name'][:30]} | {seg['title'][:40]}...")

    # Count unique episodes before reranking
    unique_episodes_before = len(set(seg['content_id'] for seg in results))
    unique_channels_before = len(set(seg['channel_name'] for seg in results))
    print(f"\n   Before reranking: {len(results)} segments, {unique_episodes_before} episodes, {unique_channels_before} channels")

    print(f"\n3. Reranking with diversity constraints...")
    t0 = time.perf_counter()

    # Reranker now uses embedded psycopg2 connection - no session needed
    reranker = SegmentReranker()

    weights = RerankerWeights(
        similarity=0.4,
        popularity=0.2,
        recency=0.2,
        single_speaker=0.1,
        named_speaker=0.1
    )

    diversity = DiversityConstraints(
        best_per_episode=True,
        max_per_channel=None  # No limit for now
    )

    reranked = reranker.rerank(
        results,
        weights=weights,
        diversity=diversity,
        time_window_days=time_window_days
    )

    t1 = time.perf_counter()
    rerank_time = (t1 - t0) * 1000

    unique_episodes_after = len(set(seg['content_id'] for seg in reranked))
    unique_channels_after = len(set(seg['channel_name'] for seg in reranked))

    print(f"   Reranked to {len(reranked)} segments in {rerank_time:.1f}ms")
    print(f"   After reranking: {len(reranked)} segments, {unique_episodes_after} episodes, {unique_channels_after} channels")

    # Show reranked results
    print(f"\n   Sample reranked results (top 10):")
    for i, seg in enumerate(reranked[:10]):
        sim = seg.get('similarity', 0)
        rerank_score = seg.get('rerank_score', 0)
        pop = seg.get('_channel_popularity', 0)
        single = seg.get('_is_single_speaker', False)
        named = seg.get('_has_named_speaker', False)

        flags = []
        if single:
            flags.append("1spk")
        if named:
            flags.append("named")
        flags_str = f" [{','.join(flags)}]" if flags else ""

        print(f"   [{i+1}] rerank={rerank_score:.3f} sim={sim:.3f} pop={pop:.2f}{flags_str}")
        print(f"       {seg['channel_name'][:35]} | {seg['title'][:45]}...")

    # Performance summary
    print(f"\n" + "=" * 60)
    print("Performance Summary")
    print("=" * 60)
    print(f"   Search time:    {search_time:.1f}ms")
    print(f"   Rerank time:    {rerank_time:.1f}ms")
    print(f"   Total time:     {search_time + rerank_time:.1f}ms")
    print(f"   Segments:       {len(results)} -> {len(reranked)} (best per episode)")
    print(f"   Episodes:       {unique_episodes_before} -> {unique_episodes_after}")
    print(f"   Channels:       {unique_channels_before} -> {unique_channels_after}")


async def test_batch_search_with_reranking():
    """Test batch search (multiple query embeddings) with reranking."""

    print("\n" + "=" * 60)
    print("Batch Search + Reranking Test")
    print("=" * 60)

    from src.backend.app.services.pgvector_search_service import PgVectorSearchService

    # Import reranker directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "segment_reranker",
        Path(__file__).parent.parent / "src/backend/app/services/rag/segment_reranker.py"
    )
    segment_reranker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(segment_reranker)
    SegmentReranker = segment_reranker.SegmentReranker
    RerankerWeights = segment_reranker.RerankerWeights
    DiversityConstraints = segment_reranker.DiversityConstraints

    class MockConfig:
        project = "CPRMV"
        allowed_projects = ["CPRMV"]
        use_alt_embeddings = False

    config = MockConfig()
    search_service = PgVectorSearchService("test_batch", config)

    # Get multiple real embeddings from the database
    print("\n1. Fetching 5 real embeddings from database...")
    t0 = time.perf_counter()

    import psycopg2
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(
        host='10.0.0.4',
        database='av_content',
        user='signal4',
        password='signal4'
    )
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    # Get embeddings from different segments to simulate query expansion diversity
    cursor.execute("""
        SELECT embedding FROM embedding_cache_30d
        WHERE embedding IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 5
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    query_embeddings = []
    for row in rows:
        emb_str = row['embedding']
        if isinstance(emb_str, str):
            emb_str = emb_str.strip('[]')
            emb = np.array([float(x) for x in emb_str.split(',')], dtype=np.float32)
        else:
            emb = np.array(emb_str, dtype=np.float32)
        query_embeddings.append(emb)

    t1 = time.perf_counter()
    print(f"   Fetched {len(query_embeddings)} embeddings in {(t1-t0)*1000:.1f}ms")

    # Batch unified search
    print("\n2. Running batch_search_unified...")
    t0 = time.perf_counter()

    results = search_service.batch_search_unified(
        query_embeddings,
        time_window_days=30,
        k=None,  # No limit - get all above threshold
        threshold=0.35
    )

    t1 = time.perf_counter()
    search_time = (t1 - t0) * 1000
    print(f"   Retrieved {len(results)} unique segments in {search_time:.1f}ms")

    if not results:
        print("   No results found.")
        return

    # Rerank
    print("\n3. Reranking...")
    t0 = time.perf_counter()

    reranker = SegmentReranker()
    reranked = reranker.rerank(
        results,
        weights=RerankerWeights(),
        diversity=DiversityConstraints(best_per_episode=True),
        time_window_days=30
    )

    t1 = time.perf_counter()
    rerank_time = (t1 - t0) * 1000

    print(f"   Reranked to {len(reranked)} segments in {rerank_time:.1f}ms")

    # Summary
    print(f"\n" + "=" * 60)
    print("Batch Performance Summary")
    print("=" * 60)
    print(f"   Queries:        {len(query_embeddings)}")
    print(f"   Search time:    {search_time:.1f}ms")
    print(f"   Rerank time:    {rerank_time:.1f}ms")
    print(f"   Total time:     {search_time + rerank_time:.1f}ms")
    print(f"   Segments:       {len(results)} -> {len(reranked)}")


async def test_fast_clustering():
    """Test fast sub-theme detection."""

    print("\n" + "=" * 60)
    print("Fast Clustering Test (Sub-theme Detection)")
    print("=" * 60)

    from src.backend.app.services.pgvector_search_service import PgVectorSearchService

    # Import fast clustering directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fast_clustering",
        Path(__file__).parent.parent / "src/backend/app/services/rag/fast_clustering.py"
    )
    fast_clustering = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fast_clustering)
    cluster_segments_fast = fast_clustering.cluster_segments_fast

    class MockConfig:
        project = "CPRMV"
        allowed_projects = ["CPRMV"]
        use_alt_embeddings = False

    config = MockConfig()
    search_service = PgVectorSearchService("test_clustering", config)

    # Get embeddings for clustering test
    print("\n1. Fetching embeddings for clustering test...")
    t0 = time.perf_counter()

    import psycopg2
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(
        host='10.0.0.4',
        database='av_content',
        user='signal4',
        password='signal4'
    )
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        SELECT embedding FROM embedding_cache_30d
        WHERE embedding IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 3
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    query_embeddings = []
    for row in rows:
        emb_str = row['embedding']
        if isinstance(emb_str, str):
            emb_str = emb_str.strip('[]')
            emb = np.array([float(x) for x in emb_str.split(',')], dtype=np.float32)
        else:
            emb = np.array(emb_str, dtype=np.float32)
        query_embeddings.append(emb)

    t1 = time.perf_counter()
    print(f"   Fetched {len(query_embeddings)} query embeddings in {(t1-t0)*1000:.1f}ms")

    # Get segments with embeddings
    print("\n2. Retrieving segments for clustering...")
    t0 = time.perf_counter()

    results = search_service.batch_search_unified(
        query_embeddings,
        time_window_days=30,
        k=None,
        threshold=0.30  # Lower threshold to get more segments
    )

    t1 = time.perf_counter()
    print(f"   Retrieved {len(results)} segments in {(t1-t0)*1000:.1f}ms")

    if len(results) < 30:
        print("   Not enough segments for clustering test. Need at least 30.")
        return

    # Run fast clustering
    print(f"\n3. Running fast clustering on {len(results)} segments...")
    t0 = time.perf_counter()

    groups, cluster_result = cluster_segments_fast(
        results,
        min_cluster_size=10,
        min_silhouette=0.15,
        max_clusters=4
    )

    t1 = time.perf_counter()
    cluster_time = (t1 - t0) * 1000

    print(f"   Clustering complete in {cluster_time:.1f}ms")
    print(f"   Clusters found: {cluster_result.n_clusters}")
    print(f"   Silhouette score: {cluster_result.silhouette_score:.3f}")
    print(f"   Valid clustering: {cluster_result.is_valid}")
    print(f"   Cluster sizes: {cluster_result.cluster_sizes}")

    # Show sample from each cluster
    if cluster_result.n_clusters >= 2:
        print(f"\n   Sample segments from each cluster:")
        for i, group in enumerate(groups):
            if group:
                seg = group[0]
                print(f"   [Cluster {i+1}] ({len(group)} segments)")
                print(f"      {seg['channel_name'][:30]} | {seg['title'][:40]}...")

    # Performance summary
    print(f"\n" + "=" * 60)
    print("Fast Clustering Summary")
    print("=" * 60)
    print(f"   Segments:       {len(results)}")
    print(f"   Cluster time:   {cluster_time:.1f}ms")
    print(f"   Clusters:       {cluster_result.n_clusters}")
    print(f"   Silhouette:     {cluster_result.silhouette_score:.3f}")
    print(f"   Valid:          {cluster_result.is_valid}")


if __name__ == "__main__":
    asyncio.run(test_reranking())
    asyncio.run(test_batch_search_with_reranking())
    asyncio.run(test_fast_clustering())
