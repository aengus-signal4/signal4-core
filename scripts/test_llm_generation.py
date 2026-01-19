#!/usr/bin/env python3
"""
Test LLM generation paths to evaluate template quality.

Tests both:
1. Single-pass (rag_answer) - consensus case
2. Two-pass (cluster_perspective_summary + rag_synthesis) - diverging views case
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment from .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np


async def test_single_pass():
    """Test single-pass generation (rag_answer template) - consensus case."""

    print("=" * 80)
    print("TEST 1: Single-Pass Generation (Consensus Case)")
    print("=" * 80)

    from src.backend.app.services.pgvector_search_service import PgVectorSearchService
    # Import TextGenerator directly to avoid umap dependency chain
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "text_generator",
        Path(__file__).parent.parent / "src/backend/app/services/rag/text_generator.py"
    )
    text_gen_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(text_gen_module)
    TextGenerator = text_gen_module.TextGenerator

    from sentence_transformers import SentenceTransformer

    # Config
    class MockConfig:
        project = "Canadian"
        allowed_projects = ["Canadian", "Big_Channels"]
        use_alt_embeddings = False

    config = MockConfig()
    search_service = PgVectorSearchService("test", config)

    # Load model
    print("\n1. Loading embedding model...")
    embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)

    # Query
    query = "What are people saying about Mark Carney"
    print(f"\n2. Query: '{query}'")

    query_embedding = embedding_model.encode(query, normalize_embeddings=True)

    # Search
    print("\n3. Searching (30 days, threshold=0.42, must contain 'Carney')...")
    results = search_service.search(
        query_embedding=query_embedding,
        time_window_days=30,
        k=100,
        threshold=0.42,
        must_contain_any=["Carney", "Mark Carney"],
        filter_projects=["Canadian", "Big_Channels"]
    )
    print(f"   Found {len(results)} segments")

    if len(results) < 5:
        print("   Not enough results for test")
        return

    # Sample 15 segments for generation (simulating what pipeline does)
    sample_size = min(15, len(results))
    sampled = results[:sample_size]

    # Format segments for prompt (exactly how pipeline does it)
    def format_segments_for_prompt(segments):
        parts = []
        for i, seg in enumerate(segments):
            citation_id = f"{{seg_{i+1}}}"
            text = seg.get('text', str(seg))
            channel = seg.get('channel_name', 'Unknown')
            date = seg.get('publish_date', 'Unknown')

            parts.append(
                f"{citation_id}\n"
                f"Channel: {channel}\n"
                f"Date: {date}\n"
                f"Text: {text}\n"
            )
        return "\n---\n".join(parts)

    segments_text = format_segments_for_prompt(sampled)

    print(f"\n4. LLM INPUT (rag_answer template):")
    print("-" * 80)
    print(f"Question: \"{query}\"")
    print(f"\nTranscripts ({sample_size} segments):")
    print(segments_text[:2000] + "..." if len(segments_text) > 2000 else segments_text)
    print("-" * 80)

    # Generate
    print("\n5. Generating response...")
    from src.backend.app.services.llm_service import LLMService

    class MinimalConfig:
        project = "test"
        llm_model = "grok-4-fast-non-reasoning-latest"

    llm_service = LLMService(MinimalConfig(), dashboard_id='test_generation')
    generator = TextGenerator(llm_service)

    t0 = time.perf_counter()
    response = await generator.generate_from_template(
        template_name="rag_answer",
        context={
            "theme_name": query,
            "segments_text": segments_text
        },
        model="grok-4-fast-non-reasoning-latest",
        temperature=0.3,
        max_tokens=600
    )
    t1 = time.perf_counter()

    print(f"\n6. LLM OUTPUT ({(t1-t0)*1000:.0f}ms):")
    print("-" * 80)
    print(response)
    print("-" * 80)

    # Analyze output quality
    citation_count = response.count("{seg_")
    word_count = len(response.split())

    print(f"\n7. Quality metrics:")
    print(f"   - Word count: {word_count}")
    print(f"   - Citation count: {citation_count}")
    print(f"   - Avg words/citation: {word_count/citation_count:.1f}" if citation_count > 0 else "   - No citations!")

    return response


async def test_two_pass():
    """Test two-pass generation (clusters) - diverging views case."""

    print("\n" + "=" * 80)
    print("TEST 2: Two-Pass Generation (Diverging Views Case)")
    print("=" * 80)

    from src.backend.app.services.pgvector_search_service import PgVectorSearchService
    # Import TextGenerator directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "text_generator",
        Path(__file__).parent.parent / "src/backend/app/services/rag/text_generator.py"
    )
    text_gen_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(text_gen_module)
    TextGenerator = text_gen_module.TextGenerator

    # Import fast_clustering directly
    spec2 = importlib.util.spec_from_file_location(
        "fast_clustering",
        Path(__file__).parent.parent / "src/backend/app/services/rag/fast_clustering.py"
    )
    fast_clustering_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(fast_clustering_module)
    cluster_segments_fast = fast_clustering_module.cluster_segments_fast

    from sentence_transformers import SentenceTransformer

    # Config
    class MockConfig:
        project = "Canadian"
        allowed_projects = ["Canadian", "Big_Channels"]
        use_alt_embeddings = False

    config = MockConfig()
    search_service = PgVectorSearchService("test", config)

    # Load model
    print("\n1. Loading embedding model...")
    embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)

    # Query
    query = "What are people saying about Mark Carney"
    print(f"\n2. Query: '{query}'")

    query_embedding = embedding_model.encode(query, normalize_embeddings=True)

    # Search - get more results for clustering
    print("\n3. Searching (30 days, k=300 for clustering)...")
    results = search_service.search(
        query_embedding=query_embedding,
        time_window_days=30,
        k=300,
        threshold=0.42,
        must_contain_any=["Carney", "Mark Carney"],
        filter_projects=["Canadian", "Big_Channels"]
    )
    print(f"   Found {len(results)} segments")

    if len(results) < 30:
        print("   Not enough results for clustering test (need 30+)")
        return

    # Cluster
    print("\n4. Fast clustering (PCA + KMeans)...")
    groups, cluster_result = cluster_segments_fast(
        results,
        min_cluster_size=8,
        min_silhouette=0.10,
        max_clusters=4
    )

    print(f"   Clusters: {cluster_result.n_clusters}")
    print(f"   Silhouette: {cluster_result.silhouette_score:.3f}")
    print(f"   Cluster sizes: {cluster_result.cluster_sizes}")

    if cluster_result.n_clusters < 2:
        print("   Not enough clusters for two-pass test")
        return

    # Format segments for prompt
    def format_segments_for_prompt(segments):
        parts = []
        for i, seg in enumerate(segments):
            citation_id = f"{{seg_{i+1}}}"
            text = seg.get('text', str(seg))
            channel = seg.get('channel_name', 'Unknown')
            date = seg.get('publish_date', 'Unknown')

            parts.append(
                f"{citation_id}\n"
                f"Channel: {channel}\n"
                f"Date: {date}\n"
                f"Text: {text}\n"
            )
        return "\n---\n".join(parts)

    from src.backend.app.services.llm_service import LLMService

    class MinimalConfig:
        project = "test"
        llm_model = "grok-4-fast-non-reasoning-latest"

    llm_service = LLMService(MinimalConfig(), dashboard_id='test_generation')
    generator = TextGenerator(llm_service)

    # PASS 1: Generate per-group summaries
    print("\n5. PASS 1 - Generating group summaries...")
    print("-" * 80)

    group_summaries = []
    for i, group in enumerate(groups[:cluster_result.n_clusters]):
        if len(group) < 5:
            continue

        # Take top 8 from each group
        group_sample = group[:8]
        segments_text = format_segments_for_prompt(group_sample)

        print(f"\n   GROUP {i+1} ({len(group)} segments, using {len(group_sample)}):")
        print(f"   Channels: {set(seg['channel_name'] for seg in group_sample)}")

        t0 = time.perf_counter()
        summary = await generator.generate_from_template(
            template_name="cluster_perspective_summary",
            context={
                "parent_query": query,
                "segments_text": segments_text
            },
            model="grok-4-fast-non-reasoning-latest",
            temperature=0.3,
            max_tokens=300
        )
        t1 = time.perf_counter()

        print(f"   Summary ({(t1-t0)*1000:.0f}ms): {summary[:200]}...")
        group_summaries.append({
            "summary": summary,
            "segments": group_sample
        })

    if len(group_summaries) < 2:
        print("\n   Not enough valid groups for synthesis")
        return group_summaries

    # PASS 2: Generate synthesis
    print("\n\n6. PASS 2 - Generating synthesis...")
    print("-" * 80)

    # Format group summaries with sources (like pipeline does)
    sections = []
    current_seg_id = 1

    for i, gs in enumerate(group_summaries):
        transcript_parts = []
        for seg in gs["segments"]:
            citation_id = f"{{seg_{current_seg_id}}}"
            text = seg.get('text', str(seg))
            channel = seg.get('channel_name', 'Unknown')
            date = seg.get('publish_date', 'Unknown')

            transcript_parts.append(
                f"{citation_id}\n"
                f"Channel: {channel}\n"
                f"Date: {date}\n"
                f"Text: {text}"
            )
            current_seg_id += 1

        transcripts_text = "\n\n---\n".join(transcript_parts)

        sections.append(f"""=== GROUP {i+1} ===
Summary: {gs["summary"]}

Source transcripts:
{transcripts_text}""")

    group_summaries_with_sources = "\n\n".join(sections)

    print(f"\n   LLM INPUT (synthesis):")
    print(f"   Theme: \"{query}\"")
    print(f"   Groups: {len(group_summaries)}")
    print(f"   Total segments: {current_seg_id - 1}")

    t0 = time.perf_counter()
    synthesis = await generator.generate_from_template(
        template_name="rag_synthesis",
        context={
            "theme_name": query,
            "group_summaries_with_sources": group_summaries_with_sources
        },
        model="grok-4-fast-non-reasoning-latest",
        temperature=0.3,
        max_tokens=800
    )
    t1 = time.perf_counter()

    print(f"\n7. LLM OUTPUT (synthesis, {(t1-t0)*1000:.0f}ms):")
    print("-" * 80)
    print(synthesis)
    print("-" * 80)

    # Analyze output quality
    citation_count = synthesis.count("{seg_")
    word_count = len(synthesis.split())

    print(f"\n8. Quality metrics:")
    print(f"   - Word count: {word_count}")
    print(f"   - Citation count: {citation_count}")
    print(f"   - Groups mentioned: {len(group_summaries)}")
    print(f"   - Avg words/citation: {word_count/citation_count:.1f}" if citation_count > 0 else "   - No citations!")

    return synthesis


async def main():
    """Run both tests."""

    print("\n" + "=" * 80)
    print("LLM GENERATION PATH TESTING")
    print("=" * 80)
    print("\nThis tests the two generation paths:")
    print("  1. Single-pass (rag_answer) - for consensus queries")
    print("  2. Two-pass (cluster + synthesis) - for diverging views")
    print()

    # Test 1: Single pass
    await test_single_pass()

    # Test 2: Two pass
    await test_two_pass()

    print("\n" + "=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
