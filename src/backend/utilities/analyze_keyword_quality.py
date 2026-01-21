#!/usr/bin/env python3
"""
Analyze the quality of keyword-retrieved segments vs primary query segments
"""
import sys
import os
import asyncio

from dotenv import load_dotenv
from src.utils.paths import get_env_path
load_dotenv(get_env_path())

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.dashboard_config import load_dashboard_config
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService


async def main():
    query = "climate change"
    strategy = "conversational"
    time_window_days = 30

    print("=" * 80)
    print("KEYWORD QUALITY ANALYSIS")
    print("=" * 80)
    print(f"\nQuery: '{query}'")
    print(f"Strategy: {strategy}")
    print(f"Time window: {time_window_days} days\n")

    config = load_dashboard_config('cprmv-practitioner')
    llm = LLMService(dashboard_id='cprmv-practitioner', config=config)
    embedding_service = EmbeddingService(config=config, dashboard_id='cprmv-practitioner')
    search_service = SearchService('cprmv-practitioner', config)

    # Generate queries with keywords
    result = llm.generate_retrieval_queries(
        query=query,
        strategy=strategy,
        n_variations=5,
        include_keywords=True
    )

    queries = result.get('queries', [])
    keyword_count = result.get('keyword_count', 0)
    primary_count = len(queries) - keyword_count

    print(f"Generated {len(queries)} items:")
    print(f"  Primary queries: {primary_count}")
    print(f"  Keywords: {keyword_count}\n")

    # Search separately for primary vs keywords
    print("=" * 80)
    print("PRIMARY QUERIES (threshold 0.40)")
    print("=" * 80)

    primary_queries = queries[:primary_count]
    primary_segments = {}

    for i, q in enumerate(primary_queries, 1):
        print(f"\n{i}. {q}")

        embeddings = await embedding_service.encode_queries([q])
        if not embeddings:
            continue

        embedding = embeddings[0]
        results = search_service.search(
            embedding,
            k=100,
            time_window_days=time_window_days,
            threshold=0.40
        )

        print(f"   Found: {len(results)} segments")
        if results:
            top3 = [f"{r.get('similarity', 0):.3f}" for r in results[:3]]
            print(f"   Top 3 sims: {top3}")

        for r in results:
            seg_id = r.get('segment_id')
            if seg_id and seg_id not in primary_segments:
                primary_segments[seg_id] = r

    print(f"\n\nPrimary queries total: {len(primary_segments)} unique segments")

    # Now keywords
    print("\n\n" + "=" * 80)
    print("KEYWORD QUERIES (threshold 0.36)")
    print("=" * 80)

    keyword_queries = queries[primary_count:]
    keyword_segments = {}

    for i, q in enumerate(keyword_queries, 1):
        print(f"\n{i}. {q}")

        embeddings = await embedding_service.encode_queries([q])
        if not embeddings:
            continue

        embedding = embeddings[0]
        results = search_service.search(
            embedding,
            k=100,
            time_window_days=time_window_days,
            threshold=0.36  # Lower threshold
        )

        print(f"   Found: {len(results)} segments")
        if results:
            top3 = [f"{r.get('similarity', 0):.3f}" for r in results[:3]]
            print(f"   Top 3 sims: {top3}")
            print(f"   Avg sim: {sum(r.get('similarity', 0) for r in results) / len(results):.3f}")

        for r in results:
            seg_id = r.get('segment_id')
            if seg_id and seg_id not in keyword_segments:
                keyword_segments[seg_id] = r

    print(f"\n\nKeyword queries total: {len(keyword_segments)} unique segments")

    # Analysis
    print("\n\n" + "=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)

    overlap = set(primary_segments.keys()) & set(keyword_segments.keys())
    only_keywords = set(keyword_segments.keys()) - set(primary_segments.keys())
    only_primary = set(primary_segments.keys()) - set(keyword_segments.keys())

    print(f"\nSegment counts:")
    print(f"  Primary only: {len(only_primary)}")
    print(f"  Keywords only: {len(only_keywords)}")
    print(f"  Overlap: {len(overlap)}")
    print(f"  Total unique: {len(primary_segments) + len(keyword_segments) - len(overlap)}")

    # Sample keyword-only segments
    print(f"\n\nSAMPLE SEGMENTS FOUND ONLY BY KEYWORDS (not by primary):")
    print("-" * 80)

    keyword_only_segs = [keyword_segments[sid] for sid in list(only_keywords)[:20]]
    keyword_only_segs.sort(key=lambda x: x.get('similarity', 0), reverse=True)

    for i, seg in enumerate(keyword_only_segs[:15], 1):
        sim = seg.get('similarity', 0)
        text = seg.get('text', '')[:150]
        channel = seg.get('channel_name', 'Unknown')

        print(f"\n{i}. Similarity: {sim:.3f}")
        print(f"   Channel: {channel}")
        print(f"   Text: {text}...")

        # Assess relevance
        climate_keywords = ['climat', 'climate', 'carbon', 'carbone', 'emission', 'réchauff', 'warming', 'greenhouse', 'serre']
        is_relevant = any(kw in text.lower() for kw in climate_keywords)
        print(f"   Relevance: {'✓ Contains climate keywords' if is_relevant else '✗ No obvious climate keywords'}")

    # Similarity distribution
    print(f"\n\nSIMILARITY SCORE DISTRIBUTION:")
    print("-" * 80)

    if keyword_only_segs:
        sims = [s.get('similarity', 0) for s in keyword_segments.values()]
        print(f"\nKeyword segments (threshold 0.36):")
        print(f"  Min: {min(sims):.3f}")
        print(f"  Max: {max(sims):.3f}")
        print(f"  Avg: {sum(sims)/len(sims):.3f}")
        print(f"  Count 0.36-0.37: {len([s for s in sims if 0.36 <= s < 0.37])}")
        print(f"  Count 0.37-0.38: {len([s for s in sims if 0.37 <= s < 0.38])}")
        print(f"  Count 0.38-0.39: {len([s for s in sims if 0.38 <= s < 0.39])}")
        print(f"  Count 0.39-0.40: {len([s for s in sims if 0.39 <= s < 0.40])}")
        print(f"  Count 0.40+: {len([s for s in sims if s >= 0.40])}")

    primary_sims = [s.get('similarity', 0) for s in primary_segments.values()]
    if primary_sims:
        print(f"\nPrimary segments (threshold 0.40):")
        print(f"  Min: {min(primary_sims):.3f}")
        print(f"  Max: {max(primary_sims):.3f}")
        print(f"  Avg: {sum(primary_sims)/len(primary_sims):.3f}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
