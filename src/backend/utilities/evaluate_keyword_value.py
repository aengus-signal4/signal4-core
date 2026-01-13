#!/usr/bin/env python3
"""
Evaluate whether keywords add value at threshold 0.40
Compare primary queries vs keywords on actual segment quality
"""
import sys
import os
import asyncio

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(__file__), '../../../.env')
load_dotenv(env_path)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config.dashboard_config import load_dashboard_config
from app.services.llm_service import LLMService
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService


async def main():
    query = "climate change"
    strategy = "conversational"
    time_window_days = 30
    threshold = 0.40

    output_file = "/tmp/keyword_evaluation_30days.txt"

    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"KEYWORD VALUE EVALUATION ({time_window_days} days, threshold {threshold})\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Query: '{query}'\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Time window: {time_window_days} days\n")
        f.write(f"Threshold: {threshold} (same for both primary and keywords)\n\n")

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

        f.write(f"Generated {len(queries)} items:\n")
        f.write(f"  Primary queries: {primary_count}\n")
        f.write(f"  Keywords: {keyword_count}\n\n")

        # Search with primary queries
        f.write("=" * 80 + "\n")
        f.write("PRIMARY QUERIES\n")
        f.write("=" * 80 + "\n\n")

        primary_queries = queries[:primary_count]
        primary_segments = {}

        for i, q in enumerate(primary_queries, 1):
            f.write(f"{i}. {q}\n")

            embeddings = await embedding_service.encode_queries([q])
            if not embeddings:
                continue

            embedding = embeddings[0]
            results = search_service.search(
                embedding,
                k=100,
                time_window_days=time_window_days,
                threshold=threshold
            )

            f.write(f"   Found: {len(results)} segments\n")
            if results:
                top3 = [f"{r.get('similarity', 0):.3f}" for r in results[:3]]
                f.write(f"   Top 3 sims: {top3}\n")

            for r in results:
                seg_id = r.get('segment_id')
                if seg_id and seg_id not in primary_segments:
                    primary_segments[seg_id] = r

            f.write("\n")

        f.write(f"Primary queries total: {len(primary_segments)} unique segments\n\n")

        # Search with keywords
        f.write("=" * 80 + "\n")
        f.write("KEYWORD QUERIES\n")
        f.write("=" * 80 + "\n\n")

        keyword_queries = queries[primary_count:]
        keyword_segments = {}

        for i, q in enumerate(keyword_queries, 1):
            f.write(f"{i}. {q}\n")

            embeddings = await embedding_service.encode_queries([q])
            if not embeddings:
                continue

            embedding = embeddings[0]
            results = search_service.search(
                embedding,
                k=100,
                time_window_days=time_window_days,
                threshold=threshold
            )

            f.write(f"   Found: {len(results)} segments\n")
            if results:
                top3 = [f"{r.get('similarity', 0):.3f}" for r in results[:3]]
                f.write(f"   Top 3 sims: {top3}\n")

            for r in results:
                seg_id = r.get('segment_id')
                if seg_id and seg_id not in keyword_segments:
                    keyword_segments[seg_id] = r

            f.write("\n")

        f.write(f"Keyword queries total: {len(keyword_segments)} unique segments\n\n")

        # Analysis
        f.write("=" * 80 + "\n")
        f.write("OVERLAP ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        primary_ids = set(primary_segments.keys())
        keyword_ids = set(keyword_segments.keys())
        overlap = primary_ids & keyword_ids
        only_primary = primary_ids - keyword_ids
        only_keywords = keyword_ids - primary_ids

        f.write(f"Segment counts:\n")
        f.write(f"  Primary only: {len(only_primary)}\n")
        f.write(f"  Keywords only: {len(only_keywords)}\n")
        f.write(f"  Overlap: {len(overlap)}\n")
        f.write(f"  Total unique: {len(primary_ids | keyword_ids)}\n\n")

        # Show all segments in each category
        f.write("=" * 80 + "\n")
        f.write("SEGMENTS FOUND ONLY BY PRIMARY QUERIES\n")
        f.write("=" * 80 + "\n\n")

        only_primary_segs = [primary_segments[sid] for sid in only_primary]
        only_primary_segs.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        for i, seg in enumerate(only_primary_segs, 1):
            sim = seg.get('similarity', 0)
            text = seg.get('text', '')
            channel = seg.get('channel_name', 'Unknown')
            title = seg.get('title', 'Unknown')

            f.write(f"{i}. Similarity: {sim:.3f}\n")
            f.write(f"   Channel: {channel}\n")
            f.write(f"   Title: {title[:80]}...\n")
            f.write(f"   Text: {text[:200]}...\n\n")

        f.write("=" * 80 + "\n")
        f.write("SEGMENTS FOUND ONLY BY KEYWORDS\n")
        f.write("=" * 80 + "\n\n")

        only_keyword_segs = [keyword_segments[sid] for sid in only_keywords]
        only_keyword_segs.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        for i, seg in enumerate(only_keyword_segs, 1):
            sim = seg.get('similarity', 0)
            text = seg.get('text', '')
            channel = seg.get('channel_name', 'Unknown')
            title = seg.get('title', 'Unknown')

            f.write(f"{i}. Similarity: {sim:.3f}\n")
            f.write(f"   Channel: {channel}\n")
            f.write(f"   Title: {title[:80]}...\n")
            f.write(f"   Text: {text[:200]}...\n\n")

        f.write("=" * 80 + "\n")
        f.write("SEGMENTS IN OVERLAP (found by both)\n")
        f.write("=" * 80 + "\n\n")

        overlap_segs = [primary_segments[sid] for sid in overlap]
        overlap_segs.sort(key=lambda x: x.get('similarity', 0), reverse=True)

        for i, seg in enumerate(overlap_segs, 1):
            sim = seg.get('similarity', 0)
            text = seg.get('text', '')
            channel = seg.get('channel_name', 'Unknown')
            title = seg.get('title', 'Unknown')

            f.write(f"{i}. Similarity: {sim:.3f}\n")
            f.write(f"   Channel: {channel}\n")
            f.write(f"   Title: {title[:80]}...\n")
            f.write(f"   Text: {text[:200]}...\n\n")

        f.write("=" * 80 + "\n")
        f.write("SUMMARY FOR CLAUDE'S EVALUATION\n")
        f.write("=" * 80 + "\n\n")

        f.write("QUESTION: Do keywords add value at threshold 0.40?\n\n")
        f.write(f"- Primary queries found {len(only_primary)} unique segments\n")
        f.write(f"- Keywords found {len(only_keywords)} unique segments\n")
        f.write(f"- Both found {len(overlap)} segments (overlap)\n\n")

        f.write("EVALUATION CRITERIA:\n")
        f.write("1. Are the 'keywords only' segments relevant to climate change?\n")
        f.write("2. Do they add meaningful coverage not in 'primary only'?\n")
        f.write("3. Is the quality comparable to 'primary only' segments?\n\n")

        f.write("If keywords only retrieves <5 relevant segments OR quality is poor,\n")
        f.write("RECOMMENDATION: Drop include_keywords feature entirely.\n\n")

    print(f"Analysis written to {output_file}")
    print(f"\nPlease read the file and provide your evaluation.")


if __name__ == "__main__":
    asyncio.run(main())
