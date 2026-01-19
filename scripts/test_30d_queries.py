#!/usr/bin/env python3
"""
Test queries against the 30d cache table for performance optimization.

Tests:
1. "What is being said about Mark Carney"
2. "What is being said about the changing world order and increased economic and political instability"
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')


def test_queries():
    """Run both test queries and measure performance."""

    print("=" * 80)
    print("30d Cache Query Performance Test")
    print("=" * 80)

    # Import after path setup
    from sentence_transformers import SentenceTransformer
    from src.backend.app.services.pgvector_search_service import PgVectorSearchService

    # Mock config for testing - Canadian and Big_Channels projects
    class MockConfig:
        project = "Canadian"
        allowed_projects = ["Canadian", "Big_Channels"]
        use_alt_embeddings = False

    config = MockConfig()
    search_service = PgVectorSearchService("test_30d", config)

    # Load embedding model
    print("\n1. Loading embedding model (Qwen/Qwen3-Embedding-0.6B)...")
    t0 = time.perf_counter()
    embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)
    model_load_time = (time.perf_counter() - t0) * 1000
    print(f"   Model loaded in {model_load_time:.0f}ms")

    # Test queries
    queries = [
        ("Mark Carney", "What is being said about Mark Carney"),
        ("World Order", "What is being said about the changing world order and increased economic and political instability"),
    ]

    results_summary = []

    for name, query_text in queries:
        print(f"\n{'='*80}")
        print(f"Query: '{name}'")
        print(f"Full text: {query_text}")
        print("-" * 80)

        # Generate embedding
        t0 = time.perf_counter()
        query_embedding = embedding_model.encode(
            f"Instruct: Retrieve relevant passages.\nQuery: {query_text}",
            normalize_embeddings=True
        )
        embed_time = (time.perf_counter() - t0) * 1000
        print(f"\n2. Embedding generated in {embed_time:.1f}ms (dim={len(query_embedding)})")

        # Run search on 30d cache
        print(f"\n3. Searching 30d cache...")
        t0 = time.perf_counter()

        results = search_service.search(
            query_embedding=query_embedding,
            time_window_days=30,
            k=200,
            threshold=0.40,
        )

        search_time = (time.perf_counter() - t0) * 1000
        print(f"   Found {len(results)} segments in {search_time:.1f}ms")

        # Show top 5 results
        if results:
            print(f"\n4. Top 5 results:")
            print("-" * 60)
            unique_channels = set()
            unique_episodes = set()

            for i, seg in enumerate(results[:5]):
                sim = seg.get('similarity', 0)
                print(f"   [{i+1}] sim={sim:.3f} | {seg['channel_name'][:40]}")
                print(f"       Title: {seg['title'][:55]}...")
                print(f"       Text: {seg['text'][:80]}...")
                print()

            # Stats
            unique_channels = set(seg['channel_name'] for seg in results)
            unique_episodes = set(seg['content_id'] for seg in results)
            avg_sim = sum(seg['similarity'] for seg in results) / len(results)

            print(f"   Stats: {len(results)} segments, {len(unique_episodes)} episodes, {len(unique_channels)} channels")
            print(f"   Similarity range: {results[-1]['similarity']:.3f} - {results[0]['similarity']:.3f} (avg: {avg_sim:.3f})")

        results_summary.append({
            'name': name,
            'embed_time': embed_time,
            'search_time': search_time,
            'total_time': embed_time + search_time,
            'num_results': len(results),
            'num_episodes': len(unique_episodes) if results else 0,
            'num_channels': len(unique_channels) if results else 0,
        })

    # Summary
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Model load time: {model_load_time:.0f}ms (one-time)")
    print()

    for r in results_summary:
        print(f"Query: {r['name']}")
        print(f"  Embedding:  {r['embed_time']:>7.1f}ms")
        print(f"  Search:     {r['search_time']:>7.1f}ms")
        print(f"  Total:      {r['total_time']:>7.1f}ms")
        print(f"  Results:    {r['num_results']} segments, {r['num_episodes']} episodes, {r['num_channels']} channels")
        print()


if __name__ == "__main__":
    test_queries()
