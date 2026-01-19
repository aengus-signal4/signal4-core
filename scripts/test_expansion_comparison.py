#!/usr/bin/env python3
"""
Compare search results with and without query expansion.

Tests:
1. Single query embedding only (no expansion)
2. Full 10-query expansion via Grok

Measures:
- Segments found
- Unique episodes
- Unique channels
- Overlap between approaches
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import requests
import json


async def test_expansion_comparison():
    """Compare expanded vs non-expanded search."""
    
    print("=" * 70)
    print("Query Expansion Comparison Test")
    print("=" * 70)
    
    from src.backend.app.services.pgvector_search_service import PgVectorSearchService
    from sentence_transformers import SentenceTransformer
    
    # Config
    class MockConfig:
        project = "Canadian"
        allowed_projects = ["Canadian", "Big_Channels"]
        use_alt_embeddings = False
    
    config = MockConfig()
    search_service = PgVectorSearchService("test_expansion", config)
    
    # Load embedding model
    print("\n0. Loading embedding model...")
    t0 = time.perf_counter()
    embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)
    t1 = time.perf_counter()
    print(f"   Model loaded in {(t1-t0)*1000:.0f}ms")
    
    # Test query
    query = "What are people saying about Mark Carney"
    keyword_filter = ["Carney", "Mark Carney"]  # OR logic
    
    print(f"\n   Query: '{query}'")
    print(f"   Keyword filter: must_contain_any={keyword_filter}")
    
    # =========================================================================
    # TEST 1: Single query embedding (no expansion)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: Single Query (No Expansion)")
    print("=" * 70)
    
    t0 = time.perf_counter()
    single_embedding = embedding_model.encode(query, normalize_embeddings=True)
    embed_time_single = (time.perf_counter() - t0) * 1000
    print(f"\n1. Embedded single query in {embed_time_single:.0f}ms")
    
    t0 = time.perf_counter()
    results_single = search_service.search(
        query_embedding=single_embedding,
        time_window_days=30,
        k=500,
        threshold=0.42,
        must_contain_any=keyword_filter,
        filter_projects=["Canadian", "Big_Channels"]
    )
    search_time_single = (time.perf_counter() - t0) * 1000
    
    segments_single = set(r['segment_id'] for r in results_single)
    episodes_single = set(r['content_id'] for r in results_single)
    channels_single = set(r['channel_name'] for r in results_single)
    
    print(f"2. Search completed in {search_time_single:.0f}ms")
    print(f"   Segments: {len(results_single)}")
    print(f"   Episodes: {len(episodes_single)}")
    print(f"   Channels: {len(channels_single)}")
    
    # =========================================================================
    # TEST 2: Expanded queries (10 variations via Grok simulation)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: Expanded Query (10 Variations)")
    print("=" * 70)
    
    # Simulate query expansion - generate variations
    # In production this would be cached or use Grok API
    expanded_queries = [
        # English variations
        f"Instruct: Retrieve relevant passages.\nQuery: {query}",
        "Instruct: Retrieve relevant passages.\nQuery: Mark Carney political views and policy positions",
        "Instruct: Retrieve relevant passages.\nQuery: Public opinion on Mark Carney Liberal leadership",
        "Instruct: Retrieve relevant passages.\nQuery: Carney economic policy and fiscal approach",
        "Instruct: Retrieve relevant passages.\nQuery: Media coverage of Mark Carney campaign",
        # French variations
        "Instruct: Retrieve relevant passages.\nQuery: Ce que les gens disent de Mark Carney",
        "Instruct: Retrieve relevant passages.\nQuery: Opinion publique sur Mark Carney chef libéral",
        "Instruct: Retrieve relevant passages.\nQuery: Positions politiques de Mark Carney",
        "Instruct: Retrieve relevant passages.\nQuery: Couverture médiatique de Carney au Canada",
        "Instruct: Retrieve relevant passages.\nQuery: Carney économie et politique fiscale"
    ]
    
    print(f"\n1. Generated {len(expanded_queries)} query variations")
    for i, q in enumerate(expanded_queries[:3]):
        print(f"   [{i+1}] {q[45:85]}...")  # Show just the query part
    print(f"   ... and {len(expanded_queries)-3} more")
    
    # Embed all variations
    t0 = time.perf_counter()
    expanded_embeddings = [
        embedding_model.encode(q, normalize_embeddings=True)
        for q in expanded_queries
    ]
    embed_time_expanded = (time.perf_counter() - t0) * 1000
    print(f"\n2. Embedded {len(expanded_embeddings)} queries in {embed_time_expanded:.0f}ms")
    
    # Batch search with all embeddings
    t0 = time.perf_counter()
    results_expanded = search_service.batch_search_unified(
        expanded_embeddings,
        time_window_days=30,
        k=None,  # No limit
        threshold=0.42,
        must_contain_any=keyword_filter,
        filter_projects=["Canadian", "Big_Channels"]
    )
    search_time_expanded = (time.perf_counter() - t0) * 1000
    
    segments_expanded = set(r['segment_id'] for r in results_expanded)
    episodes_expanded = set(r['content_id'] for r in results_expanded)
    channels_expanded = set(r['channel_name'] for r in results_expanded)
    
    print(f"3. Search completed in {search_time_expanded:.0f}ms")
    print(f"   Segments: {len(results_expanded)}")
    print(f"   Episodes: {len(episodes_expanded)}")
    print(f"   Channels: {len(channels_expanded)}")
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    # Overlap analysis
    segments_only_single = segments_single - segments_expanded
    segments_only_expanded = segments_expanded - segments_single
    segments_both = segments_single & segments_expanded
    
    episodes_only_single = episodes_single - episodes_expanded
    episodes_only_expanded = episodes_expanded - episodes_single
    episodes_both = episodes_single & episodes_expanded
    
    channels_only_single = channels_single - channels_expanded
    channels_only_expanded = channels_expanded - channels_single
    channels_both = channels_single & channels_expanded
    
    print(f"\n{'Metric':<20} {'Single':<12} {'Expanded':<12} {'Overlap':<12} {'Gain':<12}")
    print("-" * 68)
    print(f"{'Segments':<20} {len(segments_single):<12} {len(segments_expanded):<12} {len(segments_both):<12} +{len(segments_only_expanded)}")
    print(f"{'Episodes':<20} {len(episodes_single):<12} {len(episodes_expanded):<12} {len(episodes_both):<12} +{len(episodes_only_expanded)}")
    print(f"{'Channels':<20} {len(channels_single):<12} {len(channels_expanded):<12} {len(channels_both):<12} +{len(channels_only_expanded)}")
    
    print(f"\n{'Timing':<20} {'Single':<12} {'Expanded':<12} {'Overhead':<12}")
    print("-" * 56)
    print(f"{'Embedding':<20} {embed_time_single:.0f}ms{'':<7} {embed_time_expanded:.0f}ms{'':<6} +{embed_time_expanded - embed_time_single:.0f}ms")
    print(f"{'Search':<20} {search_time_single:.0f}ms{'':<7} {search_time_expanded:.0f}ms{'':<6} +{search_time_expanded - search_time_single:.0f}ms")
    total_single = embed_time_single + search_time_single
    total_expanded = embed_time_expanded + search_time_expanded
    print(f"{'Total':<20} {total_single:.0f}ms{'':<7} {total_expanded:.0f}ms{'':<6} +{total_expanded - total_single:.0f}ms")
    
    # Show unique content found only via expansion
    if segments_only_expanded:
        print(f"\n--- Sample segments found ONLY with expansion ({len(segments_only_expanded)} total) ---")
        expanded_only_results = [r for r in results_expanded if r['segment_id'] in segments_only_expanded]
        for i, seg in enumerate(expanded_only_results[:5]):
            print(f"[{i+1}] sim={seg['similarity']:.3f} | {seg['channel_name'][:30]}")
            print(f"    {seg['text'][:80]}...")
    
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    segment_gain_pct = (len(segments_only_expanded) / len(segments_single) * 100) if segments_single else 0
    episode_gain_pct = (len(episodes_only_expanded) / len(episodes_single) * 100) if episodes_single else 0
    time_overhead_pct = ((total_expanded - total_single) / total_single * 100) if total_single else 0
    
    print(f"\n   Segment gain:    +{segment_gain_pct:.1f}% more segments with expansion")
    print(f"   Episode gain:    +{episode_gain_pct:.1f}% more episodes with expansion")
    print(f"   Time overhead:   +{time_overhead_pct:.1f}% slower with expansion")
    
    if segment_gain_pct > 10:
        print(f"\n   → Expansion is WORTHWHILE: Significant additional coverage")
    elif segment_gain_pct > 5:
        print(f"\n   → Expansion is MARGINAL: Small gains, consider for important queries")
    else:
        print(f"\n   → Expansion is NOT NEEDED: Keyword filter captures most results")


if __name__ == "__main__":
    asyncio.run(test_expansion_comparison())
