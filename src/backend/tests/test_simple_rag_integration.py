"""
Integration tests for SimpleRAGWorkflow with real database and components.

These tests use:
- Real PostgreSQL database queries
- Real segment retrieval
- Real embeddings (Qwen3)
- Real pgvector search
- Real LLM generation (Grok)

Run with: pytest tests/test_simple_rag_integration.py -v --log-cli-level=INFO
"""

import pytest
import os
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.rag.workflows import SimpleRAGWorkflow
from app.services.llm_service import LLMService
from app.config.dashboard_config import DashboardConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def db_engine():
    """Create database engine for testing."""
    # Get credentials from environment or use defaults
    user = os.getenv("POSTGRES_USER", "signal4")
    password = os.getenv("POSTGRES_PASSWORD", "signal4")
    host = os.getenv("POSTGRES_HOST", "10.0.0.4")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "av_content")

    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    return create_engine(connection_string, pool_pre_ping=True)


@pytest.fixture(scope="session")
def SessionLocal(db_engine):
    """Create session factory."""
    return sessionmaker(bind=db_engine)


@pytest.fixture
def db_session(SessionLocal):
    """Create database session for each test."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="session")
def llm_service():
    """Create LLMService instance."""
    # Create proper DashboardConfig for LLMService
    config_dict = {
        "dashboard": {
            "id": "test_rag",
            "name": "RAG Integration Test"
        },
        "search": {
            "embedding_model": "qwen3-embedding-4b-hf",
            "embedding_dim": 2000,
            "max_results": 200
        },
        "llm": {
            "enabled": True,
            "model": "grok-2-1212",
            "max_sample_segments": 20
        }
    }
    config = DashboardConfig(config_dict)
    return LLMService(config, dashboard_id="test_rag")


@pytest.fixture
def workflow(llm_service, db_session):
    """Create SimpleRAGWorkflow instance."""
    return SimpleRAGWorkflow(llm_service, db_session=db_session)


# ============================================================================
# Test 1: Basic RAG with Real Segments (run mode)
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_with_real_segments(workflow, db_session):
    """Test SimpleRAG with real DB segments - run() mode."""
    # Retrieve real segments from database
    from app.services.rag.segment_retriever import SegmentRetriever

    retriever = SegmentRetriever(db_session)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)

    segments = retriever.fetch_by_filter(
        projects=["CPRMV", "Europe"],
        languages=["en"],
        date_range=(start_date, end_date)
    )

    print(f"\n[TEST] Retrieved {len(segments)} segments from database")
    assert len(segments) > 0, "Should retrieve at least some segments"

    # Run workflow
    result = await workflow.run(
        query="What is being discussed about immigration policy?",
        segments=segments[:100],  # Limit to 100 for test speed
        n_samples=20
    )

    # Verify structure
    assert "summary" in result
    assert "segment_ids" in result
    assert "segment_count" in result
    assert "samples_used" in result

    # Verify data
    assert result["summary"] is not None
    assert len(result["summary"]) > 100, "Summary should be substantial"
    assert result["segment_count"] == 100
    assert result["samples_used"] == 20
    assert len(result["segment_ids"]) == 20

    # Verify segment IDs are valid
    assert all(isinstance(sid, int) for sid in result["segment_ids"])
    assert all(sid > 0 for sid in result["segment_ids"])

    print(f"[TEST] Generated summary: {len(result['summary'])} chars")
    print(f"[TEST] Used {result['samples_used']} segments")


# ============================================================================
# Test 2: RAG with Query Expansion - Multi-Query Strategy
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_with_expansion_multi_query_real(workflow):
    """Test run_with_expansion() with multi_query strategy on real data."""
    result = await workflow.run_with_expansion(
        query="Pierre Poilievre carbon tax",
        expansion_strategy="multi_query",
        k=100,
        n_samples=20,
        time_window_days=30,
        projects=["CPRMV"],
        languages=["en"]
    )

    # Verify structure
    assert "summary" in result
    assert "segment_ids" in result
    assert "segment_count" in result
    assert "samples_used" in result
    assert "expanded_queries" in result
    assert "expansion_strategy" in result

    # Verify expansion
    assert result["expansion_strategy"] == "multi_query"
    assert len(result["expanded_queries"]) == 10, "Should generate 10 query variations"

    # Verify results
    assert result["segment_count"] > 0, "Should retrieve segments"
    assert result["samples_used"] > 0, "Should sample segments"
    assert len(result["segment_ids"]) == result["samples_used"]
    assert result["summary"] is not None
    assert len(result["summary"]) > 100

    print(f"\n[TEST] Expanded to {len(result['expanded_queries'])} queries")
    print(f"[TEST] Retrieved {result['segment_count']} segments")
    print(f"[TEST] Summary: {len(result['summary'])} chars")


# ============================================================================
# Test 3: RAG with Query Expansion - Query2Doc Strategy
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_with_expansion_query2doc_real(workflow):
    """Test run_with_expansion() with query2doc strategy on real data."""
    result = await workflow.run_with_expansion(
        query="climate change debate",
        expansion_strategy="query2doc",
        k=50,
        n_samples=15,
        time_window_days=30,
        projects=["Europe"],  # Now works with array type
        languages=["en"]
    )

    # Verify structure
    assert "summary" in result
    assert "expansion_strategy" in result
    assert "expanded_queries" in result

    # Verify expansion
    assert result["expansion_strategy"] == "query2doc"
    assert len(result["expanded_queries"]) == 1, "query2doc generates 1 expanded query"

    # Verify results
    assert result["segment_count"] > 0
    assert result["samples_used"] > 0
    assert len(result["segment_ids"]) == result["samples_used"]

    print(f"\n[TEST] Query2doc expansion: {result['expanded_queries'][0][:100]}...")
    print(f"[TEST] Retrieved {result['segment_count']} segments")


# ============================================================================
# Test 4: RAG with Quantitative Metrics
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_quantitative_metrics_real(workflow, db_session):
    """Test quantitative metrics with real data."""
    from app.services.rag.segment_retriever import SegmentRetriever

    retriever = SegmentRetriever(db_session)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)

    segments = retriever.fetch_by_filter(
        projects=["CPRMV"],
        languages=["en"],
        date_range=(start_date, end_date)
    )

    print(f"\n[TEST] Retrieved {len(segments)} segments for quantitative analysis")

    result = await workflow.run(
        query="Analyze immigration discourse",
        segments=segments[:100],
        n_samples=20,
        generate_quantitative_metrics=True,
        include_baseline_for_centrality=False,  # Skip baseline for speed
        time_window_days=30
    )

    # Verify structure
    assert "quantitative_metrics" in result

    metrics = result["quantitative_metrics"]
    assert metrics is not None

    # Verify metrics structure
    assert "total_segments" in metrics
    assert "unique_videos" in metrics
    assert "unique_channels" in metrics
    assert "temporal_distribution" in metrics

    # Verify values
    assert metrics["total_segments"] == 100
    assert metrics["unique_videos"] > 0
    assert metrics["unique_channels"] > 0

    print(f"[TEST] Metrics: {metrics['total_segments']} segments, {metrics['unique_videos']} videos, {metrics['unique_channels']} channels")


# ============================================================================
# Test 5: RAG with Different Time Windows
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_different_time_windows(workflow):
    """Test RAG with different time windows."""
    for time_window in [7, 30]:
        result = await workflow.run_with_expansion(
            query="political debate",
            expansion_strategy="multi_query",
            k=50,
            n_samples=10,
            time_window_days=time_window,
            projects=["CPRMV"],
            languages=["en"]
        )

        assert result["segment_count"] > 0, f"Should retrieve segments for {time_window}d window"
        print(f"[TEST] {time_window}d window: {result['segment_count']} segments")


# ============================================================================
# Test 6: Edge Case - Empty Results
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_empty_results(workflow):
    """Test query that returns no segments."""
    result = await workflow.run_with_expansion(
        query="xyzabc123nonexistent999topic",  # Nonsense query
        expansion_strategy="multi_query",
        k=50,
        n_samples=10,
        time_window_days=7,
        projects=["CPRMV"],  # Now works with array type
        languages=["en"]
    )

    # Should handle gracefully
    assert "summary" in result
    assert result["segment_count"] == 0 or result["segment_count"] is not None
    print(f"[TEST] Empty query result: {result['segment_count']} segments")


# ============================================================================
# Test 7: Edge Case - Few Segments
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_few_segments(workflow, db_session):
    """Test when segments < n_samples requested."""
    from app.services.rag.segment_retriever import SegmentRetriever

    retriever = SegmentRetriever(db_session)
    segments = retriever.fetch_by_filter(
        projects=["CPRMV"],  # Now works with array type
        languages=["en"]
    )

    # Use only 3 segments but request 20 samples
    result = await workflow.run(
        query="Test query",
        segments=segments[:3],
        n_samples=20
    )

    # Should use all available segments
    assert result["segment_count"] == 3
    assert result["samples_used"] <= 3
    assert len(result["segment_ids"]) <= 3
    print(f"[TEST] Few segments: requested 20, got {result['samples_used']}")


# ============================================================================
# Test 8: Segment ID Traceability
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_segment_id_traceability(workflow, db_session):
    """Verify all segment IDs can be resolved back to DB."""
    from app.services.rag.segment_retriever import SegmentRetriever

    retriever = SegmentRetriever(db_session)
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=7)

    segments = retriever.fetch_by_filter(
        projects=["CPRMV"],
        languages=["en"],
        date_range=(start_date, end_date)
    )

    result = await workflow.run(
        query="Test traceability",
        segments=segments[:50],
        n_samples=10
    )

    # Verify we can fetch each segment by ID
    segment_ids = result["segment_ids"]
    assert len(segment_ids) == 10

    # Fetch segments by returned IDs
    resolved_segments = retriever.fetch_by_ids(segment_ids)

    assert len(resolved_segments) == len(segment_ids), "All segment IDs should resolve"

    # Verify segments have proper structure
    for seg in resolved_segments:
        assert seg.id in segment_ids
        assert seg.text is not None
        assert len(seg.text) > 0
        assert seg.content_id is not None

    print(f"[TEST] Successfully resolved all {len(segment_ids)} segment IDs")


# ============================================================================
# Test 9: Multi-Language Support
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_multi_language(workflow):
    """Test RAG with French language segments."""
    result = await workflow.run_with_expansion(
        query="débat sur l'immigration",
        expansion_strategy="multi_query",
        k=50,
        n_samples=15,
        time_window_days=30,
        projects=["CPRMV", "Europe"],
        languages=["fr"]
    )

    assert result["segment_count"] > 0, "Should retrieve French segments"
    assert result["summary"] is not None
    print(f"[TEST] French query: {result['segment_count']} segments")


# ============================================================================
# Test 10: Performance Benchmarking
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_performance_benchmark(workflow):
    """Benchmark RAG workflow performance."""
    import time

    start_time = time.time()

    result = await workflow.run_with_expansion(
        query="carbon tax policy debate",
        expansion_strategy="multi_query",
        k=100,
        n_samples=20,
        time_window_days=30,
        projects=["CPRMV"],
        languages=["en"]
    )

    duration = time.time() - start_time

    print(f"\n[BENCHMARK] Total time: {duration:.2f}s")
    print(f"[BENCHMARK] Segments retrieved: {result['segment_count']}")
    print(f"[BENCHMARK] Summary length: {len(result['summary'])} chars")

    # Rough performance expectations
    assert duration < 60, "Should complete within 60 seconds"
    assert result["segment_count"] > 0
    assert result["summary"] is not None
