"""
Tests for AnalysisPipeline.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime

from app.services.rag.analysis_pipeline import (
    AnalysisPipeline,
    PipelineResult
)


@pytest.fixture
def mock_llm_service():
    """Create mock LLM service."""
    service = Mock()
    service.call_grok_async = AsyncMock(return_value="Generated summary")
    return service


@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    return Mock()


@pytest.fixture
def mock_segments():
    """Create mock segments."""
    segments = []
    for i in range(100):
        seg = Mock()
        seg.id = i
        seg.text = f"Segment {i} text about topic"
        seg.channel_name = f"Channel {i % 5}"
        seg.language = "en"
        seg.publish_date = datetime(2024, 1, i % 28 + 1)
        seg.embedding = [0.1 * i] * 768  # Mock embedding
        segments.append(seg)
    return segments


@pytest.fixture
def mock_theme():
    """Create a mock theme."""
    from app.services.rag.theme_extractor import Theme

    segments = []
    for i in range(20):
        seg = Mock()
        seg.id = i
        seg.text = f"Theme segment {i}"
        seg.channel_name = f"Channel {i % 3}"
        seg.publish_date = datetime(2024, 1, 1)
        seg.embedding = [0.1 * i] * 768
        segments.append(seg)

    return Theme(
        theme_id="theme_1",
        theme_name="Immigration",
        segments=segments,
        representative_segments=segments[:5],
        keywords=["immigration", "border", "policy"],
        embedding=[0.5] * 768,
        metadata={"cluster_score": 0.8}
    )


# ============================================================================
# Initialization Tests
# ============================================================================

def test_pipeline_initialization(mock_llm_service, mock_db_session):
    """Test pipeline initialization."""
    pipeline = AnalysisPipeline(
        "test_pipeline",
        llm_service=mock_llm_service,
        db_session=mock_db_session
    )

    assert pipeline.name == "test_pipeline"
    assert pipeline.llm_service == mock_llm_service
    assert pipeline.db_session == mock_db_session
    assert len(pipeline.steps) == 0
    assert isinstance(pipeline.context, dict)

    # Lazy-initialized components should be None
    assert pipeline._retriever is None
    assert pipeline._extractor is None
    assert pipeline._selector is None
    assert pipeline._generator is None


def test_pipeline_without_services():
    """Test pipeline can be created without services."""
    pipeline = AnalysisPipeline("minimal")

    assert pipeline.llm_service is None
    assert pipeline.db_session is None


# ============================================================================
# Fluent API Tests
# ============================================================================

def test_fluent_api_chaining(mock_llm_service, mock_db_session):
    """Test fluent API method chaining."""
    pipeline = (
        AnalysisPipeline("test", llm_service=mock_llm_service, db_session=mock_db_session)
        .retrieve_segments(projects=["Europe"])
        .extract_themes(n_clusters=5)
        .select_segments(strategy="diversity", n=10)
        .generate_summaries(template="theme_summary", max_concurrent=20)
    )

    assert len(pipeline.steps) == 4
    assert pipeline.steps[0][0] == "retrieve_segments"
    assert pipeline.steps[1][0] == "extract_themes"
    assert pipeline.steps[2][0] == "select_segments"
    assert pipeline.steps[3][0] == "generate_summaries"


def test_retrieve_segments_step():
    """Test retrieve_segments step builder."""
    pipeline = AnalysisPipeline("test")

    pipeline.retrieve_segments(
        projects=["Europe", "Canadian"],
        languages=["en", "fr"],
        date_range=(datetime(2024, 1, 1), datetime(2024, 12, 31))
    )

    assert len(pipeline.steps) == 1
    step_type, params = pipeline.steps[0]
    assert step_type == "retrieve_segments"
    assert params["projects"] == ["Europe", "Canadian"]
    assert params["languages"] == ["en", "fr"]


def test_extract_themes_step():
    """Test extract_themes step builder."""
    pipeline = AnalysisPipeline("test")

    pipeline.extract_themes(
        method="hdbscan",
        n_clusters=10,
        min_cluster_size=5
    )

    step_type, params = pipeline.steps[0]
    assert step_type == "extract_themes"
    assert params["method"] == "hdbscan"
    assert params["n_clusters"] == 10
    assert params["min_cluster_size"] == 5


def test_extract_subthemes_step():
    """Test extract_subthemes step builder."""
    pipeline = AnalysisPipeline("test")

    pipeline.extract_subthemes(
        method="kmeans",
        n_subthemes=3,
        max_concurrent=5
    )

    step_type, params = pipeline.steps[0]
    assert step_type == "extract_subthemes"
    assert params["method"] == "kmeans"
    assert params["n_subthemes"] == 3
    assert params["max_concurrent"] == 5


def test_select_segments_step():
    """Test select_segments step builder."""
    pipeline = AnalysisPipeline("test")

    pipeline.select_segments(strategy="diversity", n=15)

    step_type, params = pipeline.steps[0]
    assert step_type == "select_segments"
    assert params["strategy"] == "diversity"
    assert params["n"] == 15


def test_generate_summaries_step():
    """Test generate_summaries step builder."""
    pipeline = AnalysisPipeline("test")

    pipeline.generate_summaries(
        template="theme_summary",
        max_concurrent=20,
        level="theme"
    )

    step_type, params = pipeline.steps[0]
    assert step_type == "generate_summaries"
    assert params["template"] == "theme_summary"
    assert params["max_concurrent"] == 20
    assert params["level"] == "theme"


def test_group_by_step():
    """Test group_by step builder."""
    pipeline = AnalysisPipeline("test")

    pipeline.group_by("language")

    step_type, params = pipeline.steps[0]
    assert step_type == "group_by"
    assert params["field"] == "language"


def test_custom_step():
    """Test custom step builder."""
    async def custom_func(context, **params):
        return context

    pipeline = AnalysisPipeline("test")
    pipeline.custom_step("my_custom_step", custom_func, param1="value1")

    step_type, params = pipeline.steps[0]
    assert step_type == "custom"
    assert params["name"] == "my_custom_step"
    assert params["func"] == custom_func
    assert params["param1"] == "value1"


# ============================================================================
# Component Access Tests
# ============================================================================

def test_lazy_component_initialization(mock_llm_service, mock_db_session):
    """Test lazy initialization of components."""
    pipeline = AnalysisPipeline(
        "test",
        llm_service=mock_llm_service,
        db_session=mock_db_session
    )

    # Initially None
    assert pipeline._retriever is None
    assert pipeline._generator is None

    # Access should create them
    retriever = pipeline._get_retriever()
    assert retriever is not None
    assert pipeline._retriever is retriever  # Cached

    generator = pipeline._get_generator()
    assert generator is not None
    assert pipeline._generator is generator  # Cached

    # Subsequent access should return cached
    assert pipeline._get_retriever() is retriever
    assert pipeline._get_generator() is generator


def test_generator_requires_llm_service():
    """Test that TextGenerator requires llm_service."""
    pipeline = AnalysisPipeline("test")  # No llm_service

    with pytest.raises(RuntimeError, match="TextGenerator requires llm_service"):
        pipeline._get_generator()


# ============================================================================
# Batch Execution Tests
# ============================================================================

@pytest.mark.asyncio
async def test_execute_empty_pipeline():
    """Test executing pipeline with no steps."""
    pipeline = AnalysisPipeline("empty")

    result = await pipeline.execute()

    assert isinstance(result, PipelineResult)
    assert result.name == "empty"
    assert result.steps_completed == 0
    assert result.total_steps == 0
    assert result.data == {}


@pytest.mark.asyncio
async def test_execute_retrieve_segments(mock_db_session, mock_segments, monkeypatch):
    """Test retrieve_segments execution."""
    # Mock SegmentRetriever
    mock_retriever = Mock()
    mock_retriever.fetch_by_filter = Mock(return_value=mock_segments)

    pipeline = AnalysisPipeline("test", db_session=mock_db_session)

    # Inject mock
    pipeline._retriever = mock_retriever

    pipeline.retrieve_segments(projects=["Europe"], languages=["en"])

    result = await pipeline.execute()

    assert result.steps_completed == 1
    assert "segments" in result.data
    assert len(result.data["segments"]) == 100
    mock_retriever.fetch_by_filter.assert_called_once_with(
        projects=["Europe"],
        languages=["en"]
    )


@pytest.mark.asyncio
async def test_execute_extract_themes(mock_segments):
    """Test extract_themes execution."""
    from app.services.rag.theme_extractor import Theme

    # Mock ThemeExtractor
    mock_extractor = Mock()
    mock_themes = [
        Theme("theme_1", "Immigration", mock_segments[:30], mock_segments[:5], ["immigration"], None, {}),
        Theme("theme_2", "Economy", mock_segments[30:60], mock_segments[30:35], ["economy"], None, {})
    ]
    mock_extractor.extract_by_clustering = Mock(return_value=mock_themes)

    pipeline = AnalysisPipeline("test")
    pipeline._extractor = mock_extractor

    # Add retrieve step first to populate segments
    mock_retriever = Mock()
    mock_retriever.fetch_by_filter = Mock(return_value=mock_segments)
    pipeline._retriever = mock_retriever

    pipeline.retrieve_segments(projects=["Europe"])
    pipeline.extract_themes(method="hdbscan", n_clusters=2)

    result = await pipeline.execute()

    assert "themes" in result.data
    assert len(result.data["themes"]) == 2
    mock_extractor.extract_by_clustering.assert_called_once()


@pytest.mark.asyncio
async def test_execute_select_segments(mock_theme, mock_segments):
    """Test select_segments execution."""
    mock_selector = Mock()
    mock_selector.select = Mock(return_value=mock_theme.segments[:10])

    # Mock extractor to provide themes
    mock_extractor = Mock()
    mock_extractor.extract_by_clustering = Mock(return_value=[mock_theme])

    # Mock retriever to provide segments
    mock_retriever = Mock()
    mock_retriever.fetch_by_filter = Mock(return_value=mock_segments)

    pipeline = AnalysisPipeline("test")
    pipeline._selector = mock_selector
    pipeline._extractor = mock_extractor
    pipeline._retriever = mock_retriever

    pipeline.retrieve_segments(projects=["Europe"])
    pipeline.extract_themes(n_clusters=5)
    pipeline.select_segments(strategy="diversity", n=10)

    result = await pipeline.execute()

    # Check that theme has selected attribute
    assert "themes" in result.data
    assert hasattr(result.data["themes"][0], "selected")
    mock_selector.select.assert_called_once()


@pytest.mark.asyncio
async def test_execute_generate_summaries(mock_llm_service, mock_theme, mock_segments):
    """Test generate_summaries execution."""
    mock_generator = Mock()
    mock_generator.generate_batch = AsyncMock(return_value=["Summary for theme 1"])

    # Mock extractor and retriever
    mock_extractor = Mock()
    mock_extractor.extract_by_clustering = Mock(return_value=[mock_theme])

    mock_retriever = Mock()
    mock_retriever.fetch_by_filter = Mock(return_value=mock_segments)

    pipeline = AnalysisPipeline("test", llm_service=mock_llm_service)
    pipeline._generator = mock_generator
    pipeline._extractor = mock_extractor
    pipeline._retriever = mock_retriever

    pipeline.retrieve_segments(projects=["Europe"])
    pipeline.extract_themes(n_clusters=5)
    pipeline.generate_summaries(template="theme_summary", level="theme")

    result = await pipeline.execute()

    assert "summaries" in result.data
    assert "theme" in result.data["summaries"]
    assert len(result.data["summaries"]["theme"]) == 1
    mock_generator.generate_batch.assert_called_once()


@pytest.mark.asyncio
async def test_execute_group_by(mock_segments):
    """Test group_by execution."""
    mock_retriever = Mock()
    mock_retriever.fetch_by_filter = Mock(return_value=mock_segments)

    pipeline = AnalysisPipeline("test")
    pipeline._retriever = mock_retriever

    pipeline.retrieve_segments(projects=["Europe"])
    pipeline.group_by("language")

    result = await pipeline.execute()

    assert "groups" in result.data
    assert "en" in result.data["groups"]
    assert len(result.data["groups"]["en"]) == 100


@pytest.mark.asyncio
async def test_execute_custom_step():
    """Test custom step execution."""
    async def custom_func(context, **params):
        context["custom_result"] = params.get("value", "default")
        return context

    pipeline = AnalysisPipeline("test")
    pipeline.custom_step("my_step", custom_func, value="custom_value")

    result = await pipeline.execute()

    assert "custom_result" in result.data
    assert result.data["custom_result"] == "custom_value"


@pytest.mark.asyncio
async def test_execute_full_workflow(mock_llm_service, mock_db_session, mock_segments, mock_theme):
    """Test complete workflow execution."""
    # Mock all components
    mock_retriever = Mock()
    mock_retriever.fetch_by_filter = Mock(return_value=mock_segments)

    mock_extractor = Mock()
    mock_extractor.extract_by_clustering = Mock(return_value=[mock_theme])

    mock_selector = Mock()
    mock_selector.select = Mock(return_value=mock_theme.segments[:10])

    mock_generator = Mock()
    mock_generator.generate_batch = AsyncMock(return_value=["Theme summary"])

    pipeline = AnalysisPipeline("full_workflow", llm_service=mock_llm_service, db_session=mock_db_session)
    pipeline._retriever = mock_retriever
    pipeline._extractor = mock_extractor
    pipeline._selector = mock_selector
    pipeline._generator = mock_generator

    # Build workflow
    pipeline = (
        pipeline
        .retrieve_segments(projects=["Europe"])
        .extract_themes(n_clusters=5)
        .select_segments(strategy="diversity", n=10)
        .generate_summaries(template="theme_summary", level="theme")
    )

    result = await pipeline.execute()

    assert result.steps_completed == 4
    assert "segments" in result.data
    assert "themes" in result.data
    assert "summaries" in result.data
    assert len(result.data["summaries"]["theme"]) == 1


@pytest.mark.asyncio
async def test_execute_error_handling():
    """Test error handling during execution."""
    async def failing_func(context, **params):
        raise ValueError("Custom error")

    pipeline = AnalysisPipeline("test")
    pipeline.custom_step("failing_step", failing_func)

    with pytest.raises(ValueError, match="Custom error"):
        await pipeline.execute()


# ============================================================================
# Streaming Execution Tests
# ============================================================================

@pytest.mark.asyncio
async def test_execute_stream_empty_pipeline():
    """Test streaming execution with no steps."""
    pipeline = AnalysisPipeline("empty")

    updates = []
    async for update in pipeline.execute_stream():
        updates.append(update)

    # Should only have complete message
    assert len(updates) == 1
    assert updates[0]["type"] == "complete"


@pytest.mark.asyncio
async def test_execute_stream_progress_updates(mock_db_session, mock_segments):
    """Test streaming progress updates."""
    mock_retriever = Mock()
    mock_retriever.fetch_by_filter = Mock(return_value=mock_segments)

    pipeline = AnalysisPipeline("test", db_session=mock_db_session)
    pipeline._retriever = mock_retriever
    pipeline.retrieve_segments(projects=["Europe"])

    updates = []
    async for update in pipeline.execute_stream():
        updates.append(update)

    # Should have: progress (start) -> progress (complete) -> complete
    assert len(updates) == 3
    assert updates[0]["type"] == "progress"
    assert updates[0]["step"] == "retrieve_segments"
    assert updates[0]["progress"] == 0.0

    assert updates[1]["type"] == "progress"
    assert updates[1]["progress"] == 1.0

    assert updates[2]["type"] == "complete"


@pytest.mark.asyncio
async def test_execute_stream_extract_subthemes_partial_results(mock_segments):
    """Test streaming partial results for extract_subthemes."""
    from app.services.rag.theme_extractor import Theme

    # Create 3 main themes
    themes = [
        Theme(f"theme_{i}", f"Theme {i}", mock_segments[i*10:(i+1)*10], [], [], None, {})
        for i in range(3)
    ]

    # Mock retriever
    mock_retriever = Mock()
    mock_retriever.fetch_by_filter = Mock(return_value=mock_segments)

    # Mock extractor for theme extraction
    mock_extractor = Mock()
    mock_extractor.extract_by_clustering = Mock(return_value=themes)

    def mock_extract_subthemes(theme, **kwargs):
        # Return 2 subthemes per theme
        return [
            Theme(f"{theme.theme_id}_sub_1", f"{theme.theme_name} - Position A", theme.segments[:5], [], [], None, {}),
            Theme(f"{theme.theme_id}_sub_2", f"{theme.theme_name} - Position B", theme.segments[5:], [], [], None, {})
        ]
    mock_extractor.extract_subthemes = mock_extract_subthemes

    pipeline = AnalysisPipeline("test")
    pipeline._extractor = mock_extractor
    pipeline._retriever = mock_retriever

    pipeline.retrieve_segments(projects=["Europe"])
    pipeline.extract_themes(n_clusters=3)
    pipeline.extract_subthemes(n_subthemes=2)

    partial_updates = []
    async for update in pipeline.execute_stream():
        if update["type"] == "partial":
            partial_updates.append(update)

    # Should yield one partial per theme (3 themes)
    assert len(partial_updates) == 3

    # Check first partial
    assert partial_updates[0]["step"] == "extract_subthemes"
    assert "parent_theme_name" in partial_updates[0]["data"]
    assert "subthemes" in partial_updates[0]["data"]
    assert len(partial_updates[0]["data"]["subthemes"]) == 2
    # Progress is adjusted by total steps, so just check it exists
    assert "progress" in partial_updates[0]


@pytest.mark.asyncio
async def test_execute_stream_error_handling():
    """Test error handling in streaming mode."""
    async def failing_func(context, **params):
        raise ValueError("Stream error")

    pipeline = AnalysisPipeline("test")
    pipeline.custom_step("failing", failing_func)

    updates = []
    with pytest.raises(ValueError):
        async for update in pipeline.execute_stream():
            updates.append(update)

    # Should have progress update and error update
    assert any(u["type"] == "error" for u in updates)
    error_update = [u for u in updates if u["type"] == "error"][0]
    assert "Stream error" in error_update["error"]


# ============================================================================
# Helper Methods Tests
# ============================================================================

def test_format_segments_for_prompt(mock_segments):
    """Test segment formatting for prompts."""
    pipeline = AnalysisPipeline("test")

    formatted = pipeline._format_segments_for_prompt(mock_segments[:3])

    assert "{seg_1}" in formatted
    assert "{seg_2}" in formatted
    assert "{seg_3}" in formatted
    assert "Channel:" in formatted
    assert "Date:" in formatted
    assert "Text:" in formatted


def test_prepare_generation_tasks_theme_level(mock_theme):
    """Test generation task preparation for theme level."""
    pipeline = AnalysisPipeline("test")
    context = {"themes": [mock_theme]}

    tasks, segment_id_map = pipeline._prepare_generation_tasks(
        context,
        template="theme_summary",
        level="theme",
        params={}
    )

    assert len(tasks) == 1
    assert tasks[0]["template_name"] == "theme_summary"
    assert "theme_name" in tasks[0]["context"]
    assert tasks[0]["context"]["theme_name"] == "Immigration"
    assert "metadata" in tasks[0]
    assert tasks[0]["metadata"]["theme_id"] == "theme_1"

    # Check segment ID tracking
    assert isinstance(segment_id_map, dict)
    assert "theme_theme_1" in segment_id_map
    assert isinstance(segment_id_map["theme_theme_1"], list)


def test_prepare_generation_tasks_subtheme_level():
    """Test generation task preparation for subtheme level."""
    from app.services.rag.theme_extractor import Theme

    # Create parent theme and subthemes
    segments = [Mock() for _ in range(10)]
    for i, seg in enumerate(segments):
        seg.id = 1000 + i  # Add segment IDs

    parent = Theme("parent_1", "Immigration", segments, [], [], None, {})
    sub1 = Theme("sub_1", "Pro-restriction", segments[:5], [], [], None, {})
    sub2 = Theme("sub_2", "Pro-open-borders", segments[5:], [], [], None, {})

    pipeline = AnalysisPipeline("test")
    context = {
        "themes": [parent],
        "subtheme_map": {
            "parent_1": [sub1, sub2]
        }
    }

    tasks, segment_id_map = pipeline._prepare_generation_tasks(
        context,
        template="subtheme_summary",
        level="subtheme",
        params={}
    )

    assert len(tasks) == 2
    assert tasks[0]["context"]["parent_theme_name"] == "Immigration"
    assert tasks[0]["context"]["subtheme_name"] == "Pro-restriction"
    assert tasks[1]["context"]["subtheme_name"] == "Pro-open-borders"

    # Check segment ID tracking
    assert isinstance(segment_id_map, dict)
    assert len(segment_id_map) == 2
    # Check segment IDs are tracked for each subtheme
    for task_id, seg_ids in segment_id_map.items():
        assert isinstance(seg_ids, list)
        assert len(seg_ids) > 0


def test_prepare_generation_tasks_no_themes():
    """Test task preparation with no themes."""
    pipeline = AnalysisPipeline("test")
    context = {"themes": []}

    tasks, segment_id_map = pipeline._prepare_generation_tasks(
        context,
        template="theme_summary",
        level="theme",
        params={}
    )

    assert tasks == []
    assert segment_id_map == {}


# ============================================================================
# Utility Methods Tests
# ============================================================================

def test_describe_pipeline():
    """Test pipeline description."""
    pipeline = (
        AnalysisPipeline("test")
        .retrieve_segments(projects=["Europe"])
        .extract_themes(n_clusters=5)
    )

    description = pipeline.describe()

    assert "Pipeline 'test'" in description
    assert "retrieve_segments" in description
    assert "extract_themes" in description
    assert "projects" in description


def test_pipeline_repr():
    """Test pipeline representation."""
    pipeline = (
        AnalysisPipeline("test")
        .retrieve_segments(projects=["Europe"])
        .extract_themes(n_clusters=5)
    )

    repr_str = repr(pipeline)

    assert "AnalysisPipeline" in repr_str
    assert "test" in repr_str
    assert "2 steps" in repr_str


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_integration_theme_and_subtheme_workflow(mock_llm_service, mock_db_session, mock_segments):
    """Test integrated workflow with themes and subthemes."""
    from app.services.rag.theme_extractor import Theme

    # Mock components
    mock_retriever = Mock()
    mock_retriever.fetch_by_filter = Mock(return_value=mock_segments)

    main_theme = Theme("theme_1", "Immigration", mock_segments[:50], [], [], None, {})
    mock_extractor = Mock()
    mock_extractor.extract_by_clustering = Mock(return_value=[main_theme])

    # Mock subtheme extraction
    async def mock_extract_subthemes_batch(themes, **kwargs):
        subthemes = [
            Theme("sub_1", "Pro-restriction", themes[0].segments[:25], [], [], None, {}),
            Theme("sub_2", "Pro-open", themes[0].segments[25:], [], [], None, {})
        ]
        return {"theme_1": subthemes}
    mock_extractor.extract_subthemes_batch = mock_extract_subthemes_batch

    mock_selector = Mock()
    mock_selector.select = Mock(return_value=mock_segments[:10])

    mock_generator = Mock()
    mock_generator.generate_batch = AsyncMock(return_value=["Summary 1", "Summary 2"])

    pipeline = AnalysisPipeline("hierarchical", llm_service=mock_llm_service, db_session=mock_db_session)
    pipeline._retriever = mock_retriever
    pipeline._extractor = mock_extractor
    pipeline._selector = mock_selector
    pipeline._generator = mock_generator

    # Build hierarchical workflow
    pipeline = (
        pipeline
        .retrieve_segments(projects=["Europe"])
        .extract_themes(n_clusters=5)
        .extract_subthemes(n_subthemes=2)
        .select_segments(strategy="diversity", n=10)
        .generate_summaries(template="subtheme_summary", level="subtheme")
    )

    result = await pipeline.execute()

    assert result.steps_completed == 5
    assert "segments" in result.data
    assert "themes" in result.data
    assert "subtheme_map" in result.data
    assert "summaries" in result.data
    assert "subtheme" in result.data["summaries"]
    assert len(result.data["summaries"]["subtheme"]) == 2


@pytest.mark.asyncio
async def test_integration_streaming_with_partial_results(mock_db_session, mock_segments):
    """Test streaming with partial results at each step."""
    from app.services.rag.theme_extractor import Theme

    themes = [
        Theme(f"theme_{i}", f"Theme {i}", mock_segments[i*20:(i+1)*20], [], [], None, {})
        for i in range(2)
    ]

    mock_retriever = Mock()
    mock_retriever.fetch_by_filter = Mock(return_value=mock_segments)

    mock_extractor = Mock()
    mock_extractor.extract_by_clustering = Mock(return_value=themes)

    # Mock subtheme extraction for streaming
    def mock_extract_subthemes(theme, **kwargs):
        return [
            Theme(f"{theme.theme_id}_sub_1", f"{theme.theme_name} - A", theme.segments[:10], [], [], None, {}),
            Theme(f"{theme.theme_id}_sub_2", f"{theme.theme_name} - B", theme.segments[10:], [], [], None, {})
        ]
    mock_extractor.extract_subthemes = mock_extract_subthemes

    pipeline = AnalysisPipeline("streaming_test", db_session=mock_db_session)
    pipeline._retriever = mock_retriever
    pipeline._extractor = mock_extractor

    pipeline = (
        pipeline
        .retrieve_segments(projects=["Europe"])
        .extract_themes(n_clusters=2)
        .extract_subthemes(n_subthemes=2)
    )

    updates = []
    async for update in pipeline.execute_stream():
        updates.append(update)

    # Should have progress and partial updates
    progress_updates = [u for u in updates if u["type"] == "progress"]
    partial_updates = [u for u in updates if u["type"] == "partial"]
    complete_updates = [u for u in updates if u["type"] == "complete"]

    # Progress for each step start and complete + final complete
    assert len(progress_updates) >= 3

    # Partial results for 2 themes during subtheme extraction
    assert len(partial_updates) == 2

    # One complete message
    assert len(complete_updates) == 1
    assert "execution_time_ms" in complete_updates[0]


@pytest.mark.asyncio
async def test_component_reuse_across_steps(mock_db_session, mock_segments):
    """Test that components are reused across multiple steps."""
    from app.services.rag.theme_extractor import Theme

    theme = Theme("theme_1", "Test", mock_segments[:30], [], [], None, {})

    mock_retriever = Mock()
    mock_retriever.fetch_by_filter = Mock(return_value=mock_segments)

    mock_extractor = Mock()
    mock_extractor.extract_by_clustering = Mock(return_value=[theme])

    mock_selector = Mock()
    mock_selector.select = Mock(return_value=mock_segments[:10])

    pipeline = AnalysisPipeline("reuse_test", db_session=mock_db_session)

    # Inject mocks
    pipeline._retriever = mock_retriever
    pipeline._extractor = mock_extractor
    pipeline._selector = mock_selector

    # Build pipeline with multiple retrieval and extraction steps
    pipeline = (
        pipeline
        .retrieve_segments(projects=["Europe"])
        .extract_themes(n_clusters=5)
        .select_segments(strategy="diversity", n=10)
        .extract_themes(n_clusters=3)  # Call again
        .select_segments(strategy="centrality", n=5)  # Call again
    )

    # Get components
    extractor_1 = pipeline._get_extractor()
    selector_1 = pipeline._get_selector()
    retriever_1 = pipeline._get_retriever()

    # Execute
    await pipeline.execute()

    # Get components again
    extractor_2 = pipeline._get_extractor()
    selector_2 = pipeline._get_selector()
    retriever_2 = pipeline._get_retriever()

    # Should be same instances (reused)
    assert extractor_1 is extractor_2
    assert selector_1 is selector_2
    assert retriever_1 is retriever_2

    # Verify extract_themes and select_segments were called multiple times with same instances
    assert mock_extractor.extract_by_clustering.call_count == 2
    assert mock_selector.select.call_count == 2  # 1 theme * 2 select steps
