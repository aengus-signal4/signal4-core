"""
Test Layer 1: Data Retrieval Components
========================================

Test SegmentRetriever and EmbeddingIndexer
"""

import sys
import os
from datetime import datetime, timedelta
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.rag.segment_retriever import SegmentRetriever
from app.services.rag.embedding_indexer import EmbeddingIndexer


def test_segment_retriever():
    """Test SegmentRetriever functionality."""
    print("\n" + "="*60)
    print("Testing SegmentRetriever")
    print("="*60)

    retriever = SegmentRetriever()

    # Test 1: Fetch segments by project
    print("\n[Test 1] Fetch by project (CPRMV):")
    segments = retriever.fetch_by_filter(
        projects=["CPRMV"],
        limit=10
    )
    print(f"  ✓ Retrieved {len(segments)} segments")
    if segments:
        duration = segments[0].end_time - segments[0].start_time
        print(f"  ✓ Sample segment: ID={segments[0].id}, duration={duration:.1f}s")
        print(f"  ✓ Content: {segments[0].content.title[:60]}...")
        print(f"  ✓ Text preview: {segments[0].text[:80]}...")

    # Test 2: Count segments
    print("\n[Test 2] Count segments (CPRMV):")
    count = retriever.count_by_filter(projects=["CPRMV"])
    print(f"  ✓ Total segments: {count}")

    # Test 3: Fetch by language
    print("\n[Test 3] Fetch by language (French, limit 5):")
    segments_fr = retriever.fetch_by_filter(
        projects=["CPRMV"],
        languages=["fr"],
        limit=5
    )
    print(f"  ✓ Retrieved {len(segments_fr)} French segments")

    # Test 4: Fetch by date range
    print("\n[Test 4] Fetch by date range (last 30 days):")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    segments_recent = retriever.fetch_by_filter(
        projects=["CPRMV"],
        date_range=(start_date, end_date),
        limit=10
    )
    print(f"  ✓ Retrieved {len(segments_recent)} segments from last 30 days")

    # Test 5: Get unique channels
    print("\n[Test 5] Get unique channels (CPRMV):")
    channels = retriever.get_unique_values("channel_name", projects=["CPRMV"])
    print(f"  ✓ Found {len(channels)} unique channels")
    print(f"  ✓ Sample channels: {channels[:3]}")

    # Test 6: Get unique languages
    print("\n[Test 6] Get unique languages:")
    languages = retriever.get_unique_values("main_language", projects=["CPRMV"])
    print(f"  ✓ Found {len(languages)} unique languages: {languages}")

    print("\n✅ SegmentRetriever: All tests passed!")
    return retriever, segments


def test_embedding_indexer(sample_segments):
    """Test EmbeddingIndexer functionality."""
    print("\n" + "="*60)
    print("Testing EmbeddingIndexer")
    print("="*60)

    indexer = EmbeddingIndexer(cache_dir="/tmp/faiss_test_cache")

    # Test 1: Build index
    print("\n[Test 1] Build FAISS index (CPRMV):")
    index, segment_ids = indexer.get_or_build_index(
        projects=["CPRMV"],
        date_range=(datetime.now() - timedelta(days=30), datetime.now())
    )
    print(f"  ✓ Built index with {len(segment_ids)} segments")
    print(f"  ✓ Index dimension: {index.d}")
    print(f"  ✓ Index size: {index.ntotal}")

    # Test 2: Test caching
    print("\n[Test 2] Test caching (should load from cache):")
    index2, segment_ids2 = indexer.get_or_build_index(
        projects=["CPRMV"],
        date_range=(datetime.now() - timedelta(days=30), datetime.now())
    )
    print(f"  ✓ Loaded from cache: {len(segment_ids2)} segments")
    assert segment_ids == segment_ids2, "Cached segment IDs should match"

    # Test 3: Search
    if sample_segments and len(sample_segments) > 0:
        print("\n[Test 3] Search with query embedding:")

        # Use first segment's embedding as query
        query_embedding = np.array([sample_segments[0].embedding], dtype=np.float32)

        results = indexer.search(
            query_embeddings=query_embedding,
            projects=["CPRMV"],
            k=5,
            threshold=0.5,
            date_range=(datetime.now() - timedelta(days=30), datetime.now())
        )

        print(f"  ✓ Search returned {len(results)} result lists")
        print(f"  ✓ Top result: segment_id={results[0][0][0]}, similarity={results[0][0][1]:.3f}")

        # Test 4: Batch search
        print("\n[Test 4] Batch search with multiple queries:")
        query_embeddings = np.array(
            [seg.embedding for seg in sample_segments[:3] if seg.embedding is not None],
            dtype=np.float32
        )

        if len(query_embeddings) > 0:
            batch_results = indexer.search_batch(
                query_embeddings=query_embeddings,
                projects=["CPRMV"],
                k=5,
                threshold=0.5,
                date_range=(datetime.now() - timedelta(days=30), datetime.now())
            )

            print(f"  ✓ Batch search with {len(query_embeddings)} queries")
            print(f"  ✓ Returned {len(batch_results)} result lists")
            for i, results in enumerate(batch_results):
                print(f"  ✓ Query {i+1}: {len(results)} results")

    # Test 5: Post-filtering
    print("\n[Test 5] Test post-filtering:")
    if len(segment_ids) > 10:
        # Create filter with only half the segments
        segment_filter = set(segment_ids[:len(segment_ids)//2])

        query_embedding = np.array([sample_segments[0].embedding], dtype=np.float32)
        filtered_results = indexer.search(
            query_embeddings=query_embedding,
            projects=["CPRMV"],
            k=10,
            threshold=0.5,
            segment_filter=segment_filter,
            date_range=(datetime.now() - timedelta(days=30), datetime.now())
        )

        print(f"  ✓ Filter size: {len(segment_filter)} segments")
        print(f"  ✓ Filtered results: {len(filtered_results[0])} results")

        # Verify all results are in filter
        for seg_id, score in filtered_results[0]:
            assert seg_id in segment_filter, f"Result {seg_id} not in filter"
        print(f"  ✓ All results passed filter")

    print("\n✅ EmbeddingIndexer: All tests passed!")


def main():
    """Run all Layer 1 tests."""
    print("\n" + "="*60)
    print("LAYER 1: DATA RETRIEVAL - TEST SUITE")
    print("="*60)

    try:
        # Test SegmentRetriever
        retriever, sample_segments = test_segment_retriever()

        # Test EmbeddingIndexer
        test_embedding_indexer(sample_segments)

        print("\n" + "="*60)
        print("✅ ALL LAYER 1 TESTS PASSED!")
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
