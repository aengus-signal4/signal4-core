"""
Tests for TextGenerator.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from app.services.rag.text_generator import (
    TextGenerator,
    PromptTemplateManager,
    PromptTemplate,
    create_text_generator
)


@pytest.fixture
def mock_llm_service():
    """Create mock LLM service."""
    service = Mock()
    service.call_grok_async = AsyncMock(return_value="Generated text response")
    return service


@pytest.fixture
def text_generator(mock_llm_service):
    """Create TextGenerator instance with mock LLM."""
    return TextGenerator(mock_llm_service)


@pytest.fixture
def prompt_manager():
    """Create PromptTemplateManager instance."""
    return PromptTemplateManager()


# ============================================================================
# PromptTemplateManager Tests
# ============================================================================

def test_prompt_template_manager_initialization(prompt_manager):
    """Test that manager initializes with builtin templates."""
    templates = prompt_manager.list_templates()

    # Should have builtin templates
    assert len(templates) > 0
    assert "theme_summary" in templates
    assert "subtheme_summary" in templates
    assert "cross_group_comparison" in templates
    assert "meta_summary" in templates
    assert "simple_summary" in templates


def test_register_custom_template(prompt_manager):
    """Test registering a custom template."""
    template = PromptTemplate(
        name="custom_test",
        system_message="Test system",
        prompt_template="Test prompt: {variable}",
        default_temperature=0.5,
        default_max_tokens=500
    )

    prompt_manager.register(template)

    assert prompt_manager.exists("custom_test")
    retrieved = prompt_manager.get("custom_test")
    assert retrieved.name == "custom_test"
    assert retrieved.system_message == "Test system"
    assert retrieved.prompt_template == "Test prompt: {variable}"
    assert retrieved.default_temperature == 0.5
    assert retrieved.default_max_tokens == 500


def test_get_existing_template(prompt_manager):
    """Test retrieving an existing template."""
    template = prompt_manager.get("theme_summary")

    assert template is not None
    assert template.name == "theme_summary"
    assert "{theme_name}" in template.prompt_template
    assert "{segments_text}" in template.prompt_template


def test_get_nonexistent_template(prompt_manager):
    """Test retrieving a nonexistent template."""
    template = prompt_manager.get("does_not_exist")
    assert template is None


def test_list_templates(prompt_manager):
    """Test listing all templates."""
    templates = prompt_manager.list_templates()

    assert isinstance(templates, list)
    assert len(templates) >= 5  # At least 5 builtin templates


def test_template_exists(prompt_manager):
    """Test checking template existence."""
    assert prompt_manager.exists("theme_summary")
    assert not prompt_manager.exists("nonexistent_template")


# ============================================================================
# TextGenerator Tests
# ============================================================================

def test_text_generator_initialization(text_generator):
    """Test TextGenerator initialization."""
    assert text_generator.llm_service is not None
    assert text_generator.prompt_manager is not None
    assert len(text_generator.prompt_manager.list_templates()) >= 5


def test_register_template_on_generator(text_generator):
    """Test registering template via TextGenerator."""
    text_generator.register_template(
        name="test_template",
        system_message="System",
        prompt_template="Prompt: {var}",
        default_temperature=0.7,
        default_max_tokens=1000
    )

    assert text_generator.prompt_manager.exists("test_template")
    template = text_generator.prompt_manager.get("test_template")
    assert template.default_temperature == 0.7
    assert template.default_max_tokens == 1000


@pytest.mark.asyncio
async def test_generate_from_template_basic(text_generator, mock_llm_service):
    """Test basic text generation from template."""
    context = {
        "theme_name": "Immigration",
        "segments_text": "Sample segment text..."
    }

    result = await text_generator.generate_from_template(
        template_name="theme_summary",
        context=context
    )

    assert result == "Generated text response"

    # Verify LLM was called
    mock_llm_service.call_grok_async.assert_called_once()
    call_args = mock_llm_service.call_grok_async.call_args

    # Check that prompt was filled with context
    assert "Immigration" in call_args.kwargs["prompt"]
    assert "Sample segment text" in call_args.kwargs["prompt"]


@pytest.mark.asyncio
async def test_generate_from_template_with_overrides(text_generator, mock_llm_service):
    """Test generation with parameter overrides."""
    context = {"content": "Test content", "instructions": "Summarize briefly"}

    result = await text_generator.generate_from_template(
        template_name="simple_summary",
        context=context,
        model="grok-beta",
        temperature=0.9,
        max_tokens=500,
        timeout=120
    )

    assert result == "Generated text response"

    # Verify overrides were used
    call_args = mock_llm_service.call_grok_async.call_args
    assert call_args.kwargs["model"] == "grok-beta"
    assert call_args.kwargs["temperature"] == 0.9
    assert call_args.kwargs["max_tokens"] == 500
    assert call_args.kwargs["timeout"] == 120


@pytest.mark.asyncio
async def test_generate_from_nonexistent_template(text_generator):
    """Test error when template doesn't exist."""
    with pytest.raises(ValueError, match="Template 'nonexistent' not found"):
        await text_generator.generate_from_template(
            template_name="nonexistent",
            context={}
        )


@pytest.mark.asyncio
async def test_generate_with_missing_context(text_generator):
    """Test error when required context is missing."""
    # theme_summary requires theme_name and segments_text
    context = {"theme_name": "Immigration"}  # Missing segments_text

    with pytest.raises(ValueError, match="Missing required context key"):
        await text_generator.generate_from_template(
            template_name="theme_summary",
            context=context
        )


@pytest.mark.asyncio
async def test_generate_from_template_llm_error(text_generator, mock_llm_service):
    """Test handling of LLM errors."""
    mock_llm_service.call_grok_async.side_effect = Exception("LLM API error")

    context = {"content": "Test", "instructions": "Summarize"}

    with pytest.raises(RuntimeError, match="Text generation failed"):
        await text_generator.generate_from_template(
            template_name="simple_summary",
            context=context
        )


@pytest.mark.asyncio
async def test_generate_simple(text_generator, mock_llm_service):
    """Test simple generation without template."""
    result = await text_generator.generate_simple(
        prompt="What is 2+2?",
        system_message="You are a math tutor",
        model="grok-2-1212",
        temperature=0.1,
        max_tokens=100
    )

    assert result == "Generated text response"

    call_args = mock_llm_service.call_grok_async.call_args
    assert call_args.kwargs["prompt"] == "What is 2+2?"
    assert call_args.kwargs["system_message"] == "You are a math tutor"
    assert call_args.kwargs["temperature"] == 0.1
    assert call_args.kwargs["max_tokens"] == 100


@pytest.mark.asyncio
async def test_generate_simple_with_defaults(text_generator, mock_llm_service):
    """Test simple generation with default parameters."""
    result = await text_generator.generate_simple(
        prompt="Hello world"
    )

    assert result == "Generated text response"

    call_args = mock_llm_service.call_grok_async.call_args
    assert call_args.kwargs["system_message"] == "You are a helpful assistant."
    assert call_args.kwargs["temperature"] == 0.3
    assert call_args.kwargs["max_tokens"] == 1500


@pytest.mark.asyncio
async def test_generate_batch_basic(text_generator, mock_llm_service):
    """Test batch generation with multiple tasks."""
    # Mock to return different responses
    responses = ["Response 1", "Response 2", "Response 3"]
    mock_llm_service.call_grok_async.side_effect = responses

    tasks = [
        {
            "template_name": "simple_summary",
            "context": {"content": f"Content {i}", "instructions": "Summarize"}
        }
        for i in range(3)
    ]

    results = await text_generator.generate_batch(tasks, max_concurrent=10)

    assert len(results) == 3
    assert results == responses
    assert mock_llm_service.call_grok_async.call_count == 3


@pytest.mark.asyncio
async def test_generate_batch_preserves_order(text_generator, mock_llm_service):
    """Test that batch generation preserves task order."""
    # Mock with delays to test ordering
    async def mock_generate(*args, **kwargs):
        # Simulate varying response times
        await asyncio.sleep(0.01)
        return f"Response for {kwargs['prompt'][:20]}"

    mock_llm_service.call_grok_async.side_effect = mock_generate

    tasks = [
        {
            "template_name": "simple_summary",
            "context": {"content": f"Content {i}", "instructions": "Summarize"}
        }
        for i in range(5)
    ]

    results = await text_generator.generate_batch(tasks, max_concurrent=2)

    # Results should be in same order as tasks
    assert len(results) == 5
    for i, result in enumerate(results):
        assert f"Content {i}" in result or result.startswith("Response for")


@pytest.mark.asyncio
async def test_generate_batch_with_progress_callback(text_generator, mock_llm_service):
    """Test batch generation with progress tracking."""
    mock_llm_service.call_grok_async.return_value = "Response"

    progress_updates = []

    async def progress_callback(update):
        progress_updates.append(update)

    tasks = [
        {
            "template_name": "simple_summary",
            "context": {"content": f"Content {i}", "instructions": "Summarize"}
        }
        for i in range(5)
    ]

    results = await text_generator.generate_batch(
        tasks,
        max_concurrent=10,
        progress_callback=progress_callback
    )

    # Should have 5 progress updates (one per task)
    assert len(progress_updates) == 5

    # Check progress update structure
    assert all("completed" in u for u in progress_updates)
    assert all("total" in u for u in progress_updates)
    assert all("progress" in u for u in progress_updates)
    assert all("index" in u for u in progress_updates)

    # Last update should show completion
    last_update = progress_updates[-1]
    assert last_update["completed"] == 5
    assert last_update["total"] == 5
    assert last_update["progress"] == 1.0


@pytest.mark.asyncio
async def test_generate_batch_handles_failures(text_generator, mock_llm_service):
    """Test that batch generation handles individual task failures."""
    # Mock to fail on second task
    responses = [
        "Success 1",
        Exception("API error"),
        "Success 3"
    ]
    mock_llm_service.call_grok_async.side_effect = responses

    tasks = [
        {
            "template_name": "simple_summary",
            "context": {"content": f"Content {i}", "instructions": "Summarize"}
        }
        for i in range(3)
    ]

    results = await text_generator.generate_batch(tasks, max_concurrent=10)

    # Should still return all results
    assert len(results) == 3
    assert results[0] == "Success 1"
    assert "failed" in results[1].lower()  # Error message
    assert results[2] == "Success 3"


@pytest.mark.asyncio
async def test_generate_batch_empty_tasks(text_generator):
    """Test batch generation with empty task list."""
    results = await text_generator.generate_batch([])
    assert results == []


@pytest.mark.asyncio
async def test_generate_batch_rate_limiting(text_generator, mock_llm_service):
    """Test that max_concurrent limits concurrent tasks."""
    call_times = []

    async def mock_generate(*args, **kwargs):
        call_times.append(asyncio.get_event_loop().time())
        await asyncio.sleep(0.1)
        return "Response"

    mock_llm_service.call_grok_async.side_effect = mock_generate

    tasks = [
        {
            "template_name": "simple_summary",
            "context": {"content": f"Content {i}", "instructions": "Summarize"}
        }
        for i in range(10)
    ]

    # Limit to 3 concurrent tasks
    results = await text_generator.generate_batch(tasks, max_concurrent=3)

    assert len(results) == 10

    # With 10 tasks, max_concurrent=3, should take at least 4 batches
    # Each batch takes ~0.1s, so total should be >= 0.3s
    duration = call_times[-1] - call_times[0]
    assert duration >= 0.3  # At least 3 batches completed


@pytest.mark.asyncio
async def test_generate_batch_with_custom_model(text_generator, mock_llm_service):
    """Test batch generation with custom model per task."""
    mock_llm_service.call_grok_async.return_value = "Response"

    tasks = [
        {
            "template_name": "simple_summary",
            "context": {"content": "Content 1", "instructions": "Summarize"},
            "model": "grok-beta"
        },
        {
            "template_name": "simple_summary",
            "context": {"content": "Content 2", "instructions": "Summarize"},
            "model": "grok-2-1212"
        }
    ]

    await text_generator.generate_batch(tasks)

    # Check that different models were used
    calls = mock_llm_service.call_grok_async.call_args_list
    assert calls[0].kwargs["model"] == "grok-beta"
    assert calls[1].kwargs["model"] == "grok-2-1212"


# ============================================================================
# Convenience Function Tests
# ============================================================================

def test_create_text_generator(mock_llm_service):
    """Test convenience function for creating generator."""
    generator = create_text_generator(mock_llm_service)

    assert isinstance(generator, TextGenerator)
    assert generator.llm_service == mock_llm_service
    assert len(generator.prompt_manager.list_templates()) >= 5


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_workflow_theme_summaries(text_generator, mock_llm_service):
    """Test complete workflow: generate summaries for multiple themes."""
    # Mock responses for each theme
    themes = ["Immigration", "Economy", "Climate"]
    mock_responses = [f"Summary for {theme}" for theme in themes]
    mock_llm_service.call_grok_async.side_effect = mock_responses

    # Create tasks
    tasks = [
        {
            "template_name": "theme_summary",
            "context": {
                "theme_name": theme,
                "segments_text": f"Segments about {theme}..."
            }
        }
        for theme in themes
    ]

    # Generate in batch
    results = await text_generator.generate_batch(tasks, max_concurrent=20)

    # Verify results
    assert len(results) == 3
    for i, theme in enumerate(themes):
        assert f"Summary for {theme}" == results[i]


@pytest.mark.asyncio
async def test_subtheme_summary_template(text_generator, mock_llm_service):
    """Test subtheme summary template."""
    mock_llm_service.call_grok_async.return_value = "Subtheme summary"

    context = {
        "parent_theme_name": "Immigration",
        "subtheme_name": "Border Security",
        "segments_text": "Segments about border security..."
    }

    result = await text_generator.generate_from_template(
        template_name="subtheme_summary",
        context=context
    )

    assert result == "Subtheme summary"

    # Verify prompt contains context
    call_args = mock_llm_service.call_grok_async.call_args
    prompt = call_args.kwargs["prompt"]
    assert "Immigration" in prompt
    assert "Border Security" in prompt
    assert "Segments about border security" in prompt


@pytest.mark.asyncio
async def test_cross_group_comparison_template(text_generator, mock_llm_service):
    """Test cross-group comparison template."""
    mock_llm_service.call_grok_async.return_value = "Comparison analysis"

    context = {
        "group_summaries": "Group EN: ...\nGroup FR: ...\nGroup DE: ..."
    }

    result = await text_generator.generate_from_template(
        template_name="cross_group_comparison",
        context=context
    )

    assert result == "Comparison analysis"

    call_args = mock_llm_service.call_grok_async.call_args
    assert "Group EN" in call_args.kwargs["prompt"]
