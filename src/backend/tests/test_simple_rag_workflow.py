"""
Tests for SimpleRAGWorkflow.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from app.services.rag.workflows import SimpleRAGWorkflow


@pytest.fixture
def mock_llm_service():
    """Create mock LLM service."""
    service = Mock()
    service.call_grok_async = AsyncMock(return_value="Generated summary answer")
    return service


@pytest.fixture
def mock_segments():
    """Create mock segments with IDs."""
    segments = []
    for i in range(50):
        seg = Mock()
        seg.id = 1000 + i
        seg.text = f"Segment {i} discussing topic with important information"
        seg.start_time = i * 10.0
        seg.end_time = (i + 1) * 10.0
        seg.channel_name = f"Channel {i % 5}"
        seg.publish_date = datetime(2024, 1, 1)
        seg.embedding = [0.1 * i] * 768
        segments.append(seg)
    return segments


# ============================================================================
# Basic Workflow Tests
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_workflow_basic(mock_llm_service, mock_segments):
    """Test basic SimpleRAGWorkflow execution."""
    workflow = SimpleRAGWorkflow(mock_llm_service)

    result = await workflow.run(
        query="What is discussed in these segments?",
        segments=mock_segments[:30],
        n_samples=10
    )

    assert "summary" in result
    assert "segment_ids" in result
    assert "segment_count" in result
    assert "samples_used" in result

    # Check values
    assert result["summary"] == "Generated summary answer"
    assert result["segment_count"] == 30
    assert result["samples_used"] == 10
    assert len(result["segment_ids"]) == 10


@pytest.mark.asyncio
async def test_simple_rag_workflow_tracks_segment_ids(mock_llm_service, mock_segments):
    """Test that workflow tracks segment IDs through pipeline."""
    workflow = SimpleRAGWorkflow(mock_llm_service)

    result = await workflow.run(
        query="What is the main topic?",
        segments=mock_segments,
        n_samples=20
    )

    # Check segment IDs
    assert len(result["segment_ids"]) == 20
    assert all(isinstance(sid, int) for sid in result["segment_ids"])
    assert all(1000 <= sid < 1050 for sid in result["segment_ids"])  # Valid range


@pytest.mark.asyncio
async def test_simple_rag_workflow_empty_segments(mock_llm_service):
    """Test workflow with empty segments list."""
    workflow = SimpleRAGWorkflow(mock_llm_service)

    result = await workflow.run(
        query="What is discussed?",
        segments=[],
        n_samples=10
    )

    assert result["summary"] is None
    assert result["segment_ids"] == []
    assert result["segment_count"] == 0
    assert result["samples_used"] == 0


@pytest.mark.asyncio
async def test_simple_rag_workflow_fewer_segments_than_samples(mock_llm_service, mock_segments):
    """Test workflow when fewer segments available than requested samples."""
    workflow = SimpleRAGWorkflow(mock_llm_service)

    result = await workflow.run(
        query="What is discussed?",
        segments=mock_segments[:5],
        n_samples=20  # Request more than available
    )

    # Should use all available segments
    assert result["segment_count"] == 5
    assert result["samples_used"] <= 5
    assert len(result["segment_ids"]) <= 5


# ============================================================================
# Parameter Tests
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_workflow_custom_parameters(mock_llm_service, mock_segments):
    """Test workflow with custom generation parameters."""
    workflow = SimpleRAGWorkflow(mock_llm_service)

    result = await workflow.run(
        query="Analyze this",
        segments=mock_segments,
        n_samples=15,
        diversity_weight=0.3,  # Lower diversity
        model="grok-beta",
        temperature=0.7,
        max_tokens=1000
    )

    assert result["summary"] is not None
    assert len(result["segment_ids"]) == 15

    # Verify LLM was called with custom parameters
    mock_llm_service.call_grok_async.assert_called()
    call_args = mock_llm_service.call_grok_async.call_args
    assert call_args.kwargs["model"] == "grok-beta"
    assert call_args.kwargs["temperature"] == 0.7
    assert call_args.kwargs["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_simple_rag_workflow_diversity_strategy(mock_llm_service, mock_segments):
    """Test workflow uses diversity strategy."""
    workflow = SimpleRAGWorkflow(mock_llm_service)

    # High diversity weight should use diversity strategy
    result_diverse = await workflow.run(
        query="Test",
        segments=mock_segments,
        n_samples=10,
        diversity_weight=0.9
    )

    assert len(result_diverse["segment_ids"]) == 10

    # Low diversity weight should use balanced strategy
    result_balanced = await workflow.run(
        query="Test",
        segments=mock_segments,
        n_samples=10,
        diversity_weight=0.3
    )

    assert len(result_balanced["segment_ids"]) == 10


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_workflow_full_integration(mock_llm_service, mock_segments):
    """Test complete workflow with realistic parameters."""
    workflow = SimpleRAGWorkflow(mock_llm_service)

    query = "What are the main arguments about climate policy?"

    result = await workflow.run(
        query=query,
        segments=mock_segments,
        n_samples=20,
        diversity_weight=0.8,
        model="grok-2-1212",
        temperature=0.3,
        max_tokens=500
    )

    # Verify structure
    assert isinstance(result, dict)
    assert all(key in result for key in ["summary", "segment_ids", "segment_count", "samples_used"])

    # Verify data types
    assert isinstance(result["summary"], str)
    assert isinstance(result["segment_ids"], list)
    assert isinstance(result["segment_count"], int)
    assert isinstance(result["samples_used"], int)

    # Verify values
    assert result["summary"]  # Non-empty
    assert result["segment_count"] == 50
    assert result["samples_used"] == 20
    assert len(result["segment_ids"]) == 20

    # Verify segment IDs are unique
    assert len(set(result["segment_ids"])) == len(result["segment_ids"])


@pytest.mark.asyncio
async def test_simple_rag_workflow_preserves_segment_metadata(mock_llm_service, mock_segments):
    """Test that segment IDs can be used to look up original segments."""
    workflow = SimpleRAGWorkflow(mock_llm_service)

    result = await workflow.run(
        query="Test query",
        segments=mock_segments,
        n_samples=10
    )

    # Create lookup map
    segment_map = {seg.id: seg for seg in mock_segments}

    # Verify we can resolve all segment IDs
    for seg_id in result["segment_ids"]:
        assert seg_id in segment_map
        original_seg = segment_map[seg_id]
        assert original_seg.text  # Has text
        assert original_seg.start_time >= 0  # Has timing
        assert original_seg.channel_name  # Has metadata


# ============================================================================
# Quantitative Metrics Tests
# ============================================================================

@pytest.mark.asyncio
async def test_simple_rag_workflow_quantitative_metrics(mock_llm_service, mock_segments):
    """Test workflow with quantitative metrics generation."""
    # Fix segment dates for sorting in quantitative analyzer
    for i, seg in enumerate(mock_segments[:30]):
        seg.content = Mock()
        seg.content.publish_date = datetime(2024, 1, 1 + (i % 28))  # Valid dates

    mock_db_session = Mock()
    workflow = SimpleRAGWorkflow(mock_llm_service, db_session=mock_db_session)

    result = await workflow.run(
        query="Analyze this",
        segments=mock_segments[:30],
        n_samples=10,
        generate_quantitative_metrics=True,
        include_baseline_for_centrality=False,  # Set to False to avoid DB query
        time_window_days=30
    )

    assert result["summary"] is not None
    assert "quantitative_metrics" in result
    # Note: With mocked components, metrics will be present but basic
    # Full validation happens in integration tests with real DB


@pytest.mark.asyncio
async def test_simple_rag_workflow_without_quantitative_metrics(mock_llm_service, mock_segments):
    """Test workflow WITHOUT quantitative metrics (default behavior)."""
    workflow = SimpleRAGWorkflow(mock_llm_service)

    result = await workflow.run(
        query="Analyze this",
        segments=mock_segments[:30],
        n_samples=10,
        generate_quantitative_metrics=False
    )

    assert result["summary"] is not None
    assert "quantitative_metrics" not in result  # Should not be present


# ============================================================================
# Query Expansion Tests (run_with_expansion)
# ============================================================================

@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    return Mock()


@pytest.mark.asyncio
async def test_simple_rag_workflow_with_expansion_multi_query(mock_llm_service, mock_db_session, mock_segments, monkeypatch):
    """Test run_with_expansion() with multi_query strategy - full pipeline."""
    # Setup LLM service mocks
    mock_llm_service.optimize_search_query = Mock(return_value={
        "query_variations": [f"Query variation {i}" for i in range(10)],
        "keywords": ["carbon", "tax", "debate"]
    })
    mock_llm_service.batch_convert_queries_to_embeddings = AsyncMock(
        return_value=[[0.1] * 1024 for _ in range(10)]
    )

    # Mock SearchService
    from app.services.search_service import SearchService
    mock_search_service = Mock()
    # batch_search returns a list of result lists (one per query embedding)
    mock_search_service.batch_search = Mock(return_value=[mock_segments[:5] for _ in range(10)])

    def mock_search_service_init(self, db_session, config=None):
        return None

    monkeypatch.setattr(SearchService, "__init__", mock_search_service_init)
    monkeypatch.setattr(SearchService, "batch_search", mock_search_service.batch_search)

    workflow = SimpleRAGWorkflow(mock_llm_service, db_session=mock_db_session)

    result = await workflow.run_with_expansion(
        query="carbon tax debate",
        expansion_strategy="multi_query",
        k=100,
        n_samples=20,
        time_window_days=30,
        projects=["CPRMV"]
    )

    assert "summary" in result
    assert "segment_ids" in result
    assert "segment_count" in result
    assert "samples_used" in result
    assert "expanded_queries" in result
    assert "expansion_strategy" in result

    assert result["expansion_strategy"] == "multi_query"
    assert len(result["expanded_queries"]) == 10
    # Deduplication: 10 queries × 5 segments each = 5 unique (same mocks reused)
    assert result["segment_count"] == 5
    assert result["samples_used"] == 5  # Can't sample more than available
    assert len(result["segment_ids"]) == 5


@pytest.mark.asyncio
async def test_simple_rag_workflow_with_expansion_query2doc(mock_llm_service, mock_db_session, mock_segments, monkeypatch):
    """Test run_with_expansion() with query2doc strategy."""
    # Setup LLM service mocks
    mock_llm_service.query2doc = Mock(return_value="Pseudo-document about immigration policy")
    mock_llm_service.batch_convert_queries_to_embeddings = AsyncMock(
        return_value=[[0.1] * 1024]
    )

    # Mock SearchService
    from app.services.search_service import SearchService
    mock_search_service = Mock()
    # batch_search returns a list of result lists (one per query embedding)
    mock_search_service.batch_search = Mock(return_value=[mock_segments[:30]])

    def mock_search_service_init(self, db_session, config=None):
        return None

    monkeypatch.setattr(SearchService, "__init__", mock_search_service_init)
    monkeypatch.setattr(SearchService, "batch_search", mock_search_service.batch_search)

    workflow = SimpleRAGWorkflow(mock_llm_service, db_session=mock_db_session)

    result = await workflow.run_with_expansion(
        query="immigration policy",
        expansion_strategy="query2doc",
        k=50,
        n_samples=15,
        time_window_days=30
    )

    assert "summary" in result
    assert "segment_ids" in result
    assert "expansion_strategy" in result
    assert "expanded_queries" in result

    assert result["expansion_strategy"] == "query2doc"
    assert len(result["expanded_queries"]) == 1
    assert result["segment_count"] == 30
    assert result["samples_used"] == 15
    assert len(result["segment_ids"]) == 15


@pytest.mark.asyncio
async def test_simple_rag_workflow_expansion_requires_db_session(mock_llm_service):
    """Test that run_with_expansion() requires db_session."""
    workflow = SimpleRAGWorkflow(mock_llm_service, db_session=None)

    with pytest.raises(ValueError, match="db_session required"):
        await workflow.run_with_expansion(
            query="test query",
            expansion_strategy="multi_query"
        )
