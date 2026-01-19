#!/usr/bin/env python3
"""Test hybrid search for Mark Carney query."""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


async def test_carney_query():
    """Test: What are people saying about Carney - Canadian + Big_Channels."""
    
    print("=" * 70)
    print("Hybrid Search Test: 'What are people saying about Carney'")
    print("Filters: Canadian + Big_Channels projects")
    print("=" * 70)
    
    from src.backend.app.services.pgvector_search_service import PgVectorSearchService
    from sentence_transformers import SentenceTransformer
    
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
    
    # Config for Canadian + Big_Channels
    class MockConfig:
        project = "Canadian"
        allowed_projects = ["Canadian", "Big_Channels"]
        use_alt_embeddings = False
    
    config = MockConfig()
    search_service = PgVectorSearchService("test_carney", config)
    
    # Load embedding model (same as used in production - 1024 dim)
    print("\n0. Loading embedding model...")
    t0 = time.perf_counter()
    embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)
    t1 = time.perf_counter()
    print(f"   Model loaded in {(t1-t0)*1000:.0f}ms")
    
    # The query - semantic meaning
    query = "What are people saying about Mark Carney"
    
    print(f"\n1. Embedding query: '{query}'")
    t0 = time.perf_counter()
    
    query_embedding = embedding_model.encode(query, normalize_embeddings=True)
    
    t1 = time.perf_counter()
    print(f"   Embedding ready in {(t1-t0)*1000:.1f}ms (dim={len(query_embedding)})")
    
    # Search with keyword filter for "Carney"
    print(f"\n2. Hybrid search (semantic + must contain 'Carney')...")
    print(f"   Projects: Canadian, Big_Channels")
    t0 = time.perf_counter()
    
    results = search_service.search(
        query_embedding=query_embedding,
        time_window_days=30,
        k=500,
        threshold=0.42,
        must_contain_any=["Carney", "Mark Carney"],  # OR logic for name variants
        filter_projects=["Canadian", "Big_Channels"]
    )
    
    t1 = time.perf_counter()
    search_time = (t1 - t0) * 1000
    print(f"   Found {len(results)} segments in {search_time:.1f}ms")
    
    if not results:
        print("   No results found. Trying without project filter...")
        results = search_service.search(
            query_embedding=query_embedding,
            time_window_days=30,
            k=500,
            threshold=0.42,
            must_contain_any=["Carney", "Mark Carney"]
        )
        print(f"   No project filter: {len(results)} segments")
    
    if not results:
        print("   No results with keyword filter. Trying pure semantic...")
        results = search_service.search(
            query_embedding=query_embedding,
            time_window_days=30,
            k=100,
            threshold=0.42
        )
        print(f"   Pure semantic: {len(results)} segments")
        if results:
            print("\n   Sample (pure semantic, no Carney filter):")
            for i, seg in enumerate(results[:5]):
                print(f"   [{i+1}] sim={seg['similarity']:.3f} | {seg['channel_name'][:30]}")
                print(f"       {seg['text'][:80]}...")
        return
    
    # Show raw results before reranking
    print(f"\n   Raw results (top 5 by similarity):")
    for i, seg in enumerate(results[:5]):
        print(f"   [{i+1}] sim={seg['similarity']:.3f} | {seg['channel_name'][:35]}")
        print(f"       {seg['text'][:80]}...")
    
    # Count stats before reranking
    unique_episodes = len(set(seg['content_id'] for seg in results))
    unique_channels = len(set(seg['channel_name'] for seg in results))
    print(f"\n   Before reranking: {len(results)} segments, {unique_episodes} episodes, {unique_channels} channels")
    
    # Rerank with emphasis on popularity (big channels)
    print(f"\n3. Reranking (emphasis on channel popularity)...")
    t0 = time.perf_counter()
    
    reranker = SegmentReranker()
    
    # Higher popularity weight for "big channels"
    weights = RerankerWeights(
        similarity=0.35,
        popularity=0.30,  # Higher weight for big channels
        recency=0.20,
        single_speaker=0.08,
        named_speaker=0.07
    )
    
    diversity = DiversityConstraints(
        best_per_episode=True,
        max_per_channel=3  # Limit per channel for diversity
    )
    
    reranked = reranker.rerank(
        results,
        weights=weights,
        diversity=diversity,
        time_window_days=30
    )
    
    t1 = time.perf_counter()
    rerank_time = (t1 - t0) * 1000
    
    unique_episodes_after = len(set(seg['content_id'] for seg in reranked))
    unique_channels_after = len(set(seg['channel_name'] for seg in reranked))
    
    print(f"   Reranked to {len(reranked)} segments in {rerank_time:.1f}ms")
    print(f"   After: {unique_episodes_after} episodes, {unique_channels_after} channels")
    
    # Show reranked results
    print(f"\n4. Top 15 reranked results:")
    print("-" * 70)
    
    for i, seg in enumerate(reranked[:15]):
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
        
        print(f"[{i+1}] rerank={rerank_score:.3f} sim={sim:.3f} pop={pop:.2f}{flags_str}")
        print(f"    Channel: {seg['channel_name']}")
        print(f"    Title: {seg['title'][:60]}...")
        print(f"    Text: {seg['text'][:100]}...")
        print()
    
    # Performance summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"   Query:          '{query}'")
    print(f"   Keyword filter: must_contain_any=['Carney', 'Mark Carney']")
    print(f"   Projects:       Canadian, Big_Channels")
    print(f"   Search time:    {search_time:.1f}ms")
    print(f"   Rerank time:    {rerank_time:.1f}ms")
    print(f"   Total time:     {search_time + rerank_time:.1f}ms")
    print(f"   Results:        {len(results)} -> {len(reranked)} (after diversity)")


if __name__ == "__main__":
    asyncio.run(test_carney_query())
