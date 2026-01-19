"""
RAG System Protocol Interfaces
==============================

Protocol definitions for the RAG workflow system using typing.Protocol (PEP 544).

These protocols formalize the contracts between components, enabling:
- Structural subtyping (duck typing with type checker support)
- Clear documentation of expected interfaces
- Easy creation of alternative implementations

Usage:
    from services.rag.interfaces import WorkflowProtocol, SegmentRetrieverProtocol

    def execute_workflow(workflow: WorkflowProtocol, **kwargs):
        return await workflow.run(**kwargs)
"""

from typing import (
    Protocol, TypedDict, Dict, Any, List, Optional,
    AsyncGenerator, Tuple, Callable, Union, runtime_checkable
)
from datetime import datetime
import numpy as np


# =============================================================================
# TypedDicts for Result Formats
# =============================================================================

class WorkflowEvent(TypedDict, total=False):
    """Standard event emitted during workflow execution."""
    type: str  # 'step_start', 'step_progress', 'step_complete', 'result', 'complete', 'error'
    step: str  # Step name
    progress: int  # Current progress
    total: int  # Total items
    data: Dict[str, Any]  # Step result data
    error: str  # Error message (for type='error')


class SimpleRAGResult(TypedDict, total=False):
    """Result from SimpleRAGWorkflow."""
    summary: Optional[str]
    segment_ids: List[int]
    segment_count: int
    samples_used: int
    expanded_queries: List[str]
    expansion_strategy: str
    search_results: List[Dict[str, Any]]
    quantitative_metrics: Optional[Dict[str, Any]]


class HierarchicalResult(TypedDict, total=False):
    """Result from HierarchicalSummaryWorkflow."""
    theme_summaries: List[Dict[str, Any]]
    segment_ids_by_theme: Dict[str, List[int]]
    group_results: Dict[str, Any]
    total_themes: int
    total_segments: int
    quantitative_metrics: Optional[Dict[str, Any]]


class ThemeData(TypedDict, total=False):
    """Theme data structure for serialization."""
    theme_id: str
    theme_name: str
    keywords: List[str]
    segment_count: int
    representative_text: str
    parent_theme_id: Optional[str]
    depth: int


class SegmentData(TypedDict, total=False):
    """Segment data structure for API responses."""
    segment_id: int
    text: str
    similarity: float
    content_id: int
    content_id_string: str
    channel_url: str
    channel_name: str
    title: str
    publish_date: Optional[str]
    start_time: float
    end_time: float
    speaker_hashes: List[str]
    segment_index: int
    stitch_version: Optional[str]


class QuantitativeMetrics(TypedDict, total=False):
    """Quantitative analysis result."""
    total_segments: int
    unique_videos: int
    unique_channels: int
    channel_distribution: List[Dict[str, Any]]
    video_distribution: List[Dict[str, Any]]
    temporal_distribution: Dict[str, int]
    discourse_centrality: float
    concentration_metrics: Dict[str, float]


# =============================================================================
# Workflow Protocols
# =============================================================================

@runtime_checkable
class WorkflowProtocol(Protocol):
    """
    Base protocol for workflow classes.

    Workflows orchestrate pipeline components to perform analysis tasks.
    All workflows must have an async run() method that returns a result dict.

    Attributes:
        llm_service: LLM service for text generation
        db_session: Optional database session for queries
    """
    llm_service: Any
    db_session: Optional[Any]

    async def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Execute the workflow.

        Args:
            **kwargs: Workflow-specific parameters

        Returns:
            Dict containing workflow results
        """
        ...


class StreamingWorkflowProtocol(WorkflowProtocol, Protocol):
    """
    Protocol for workflows with streaming support.

    Extends WorkflowProtocol with an async generator method for
    streaming progress events during execution.
    """

    def run_stream(self, **kwargs: Any) -> AsyncGenerator[WorkflowEvent, None]:
        """
        Execute workflow with streaming progress events.

        Args:
            **kwargs: Workflow-specific parameters

        Yields:
            WorkflowEvent dicts with progress updates
        """
        ...


# =============================================================================
# Component Protocols
# =============================================================================

class SegmentRetrieverProtocol(Protocol):
    """
    Protocol for segment retrieval components.

    Retrieves segments from the database based on various filter criteria.
    """

    def fetch_by_filter(
        self,
        projects: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        channel_urls: Optional[List[str]] = None,
        content_ids: Optional[List[str]] = None,
        segment_ids: Optional[List[int]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        must_be_stitched: bool = True,
        must_be_embedded: bool = True,
        limit: Optional[int] = None
    ) -> List[Any]:
        """
        Fetch segments matching all specified filters.

        Args:
            projects: Filter by project names
            languages: Filter by language codes
            channels: Filter by channel names
            channel_urls: Filter by channel URLs
            content_ids: Filter by content IDs
            segment_ids: Filter by specific segment IDs
            date_range: Tuple of (start_date, end_date)
            must_be_stitched: Only return stitched segments
            must_be_embedded: Only return embedded segments
            limit: Maximum segments to return

        Returns:
            List of Segment objects
        """
        ...


class ThemeExtractorProtocol(Protocol):
    """
    Protocol for theme extraction components.

    Extracts themes from segments using clustering or other strategies.
    """

    def extract_by_clustering(
        self,
        segments: List[Any],
        method: str = "hdbscan",
        n_clusters: Optional[int] = None,
        min_cluster_size: int = 5,
        max_themes: Optional[int] = None,
        use_faiss: bool = False
    ) -> List[Any]:
        """
        Extract themes using clustering.

        Args:
            segments: Segments to cluster
            method: Clustering method (hdbscan, kmeans, agglomerative)
            n_clusters: Number of clusters (required for kmeans)
            min_cluster_size: Minimum cluster size for HDBSCAN
            max_themes: Maximum number of themes to return
            use_faiss: Use FAISS for accelerated clustering

        Returns:
            List of Theme objects
        """
        ...


class SegmentSelectorProtocol(Protocol):
    """
    Protocol for segment selection components.

    Selects representative segments using weighted scoring strategies.
    """

    def select(
        self,
        segments: List[Any],
        n: int = 10,
        strategy: Optional[str] = None,
        rank_only: bool = False
    ) -> List[Any]:
        """
        Select representative segments.

        Args:
            segments: Segments to select from
            n: Number of segments to select
            strategy: Selection strategy (diversity, balanced, recency)
            rank_only: Only rank, don't limit to n

        Returns:
            List of selected Segment objects
        """
        ...


class TextGeneratorProtocol(Protocol):
    """
    Protocol for text generation components.

    Generates text using LLM with prompt templates.
    """

    async def generate(
        self,
        template_name: str,
        variables: Dict[str, Any],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate text using a prompt template.

        Args:
            template_name: Name of the template to use
            variables: Variables to substitute in template
            model: LLM model to use
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        ...

    async def generate_batch(
        self,
        tasks: List[Dict[str, Any]],
        max_concurrent: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[str]:
        """
        Generate text for multiple tasks in parallel.

        Args:
            tasks: List of task dicts with template_name and variables
            max_concurrent: Maximum concurrent generations
            progress_callback: Optional callback(completed, total)

        Returns:
            List of generated texts
        """
        ...


class QuantitativeAnalyzerProtocol(Protocol):
    """
    Protocol for quantitative analysis components.

    Analyzes segment distributions and discourse centrality metrics.
    """

    def analyze(
        self,
        segments: List[Any],
        baseline_segments: Optional[List[Any]] = None,
        time_window_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform quantitative analysis on segments.

        Args:
            segments: Segments to analyze
            baseline_segments: Optional baseline for comparison
            time_window_days: Time window for context

        Returns:
            Dict with quantitative metrics
        """
        ...


# =============================================================================
# Service Protocols
# =============================================================================

class LLMServiceProtocol(Protocol):
    """
    Protocol for LLM service.

    Provides text generation capabilities via external LLM APIs.
    """

    async def generate_async(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate text asynchronously.

        Args:
            prompt: User prompt
            system_message: System message for context
            model: Model to use
            temperature: Generation temperature
            max_tokens: Maximum tokens

        Returns:
            Generated text
        """
        ...


class EmbeddingServiceProtocol(Protocol):
    """
    Protocol for embedding service.

    Converts text to embedding vectors for semantic search.
    """

    async def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        ...

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embedding vectors, shape (len(texts), embedding_dim)
        """
        ...


# =============================================================================
# Pipeline Protocol
# =============================================================================

class PipelineStepProtocol(Protocol):
    """
    Protocol for pipeline step metadata.

    Used by step_registry.py to describe available pipeline steps.
    """
    name: str
    description: str
    parameters: Dict[str, Any]
    method_name: str


class AnalysisPipelineProtocol(Protocol):
    """
    Protocol for the analysis pipeline.

    Provides a fluent API for building and executing analysis workflows.
    """

    def execute(self) -> AsyncGenerator[WorkflowEvent, None]:
        """
        Execute the pipeline with streaming.

        Yields:
            WorkflowEvent dicts with progress updates
        """
        ...

    def custom_step(
        self,
        name: str,
        func: Callable[..., Any],
        **kwargs: Any
    ) -> "AnalysisPipelineProtocol":
        """
        Add a custom step to the pipeline.

        Args:
            name: Step name
            func: Step function (sync or async)
            **kwargs: Arguments to pass to func

        Returns:
            Self for method chaining
        """
        ...


# =============================================================================
# Type Aliases
# =============================================================================

# Segment can be either a dict (from search) or an ORM object
Segment = Any

# Theme object from theme_extractor
Theme = Any

# Pipeline context passed between steps
PipelineContext = Dict[str, Any]
