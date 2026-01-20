"""
Request Models
==============

Pydantic models for API request validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SearchRequest(BaseModel):
    """Request for semantic search"""
    query: str = Field(..., description="Search query text")
    dashboard_id: str = Field("cprmv-practitioner", description="Dashboard ID for project filtering")
    time_window_days: int = Field(7, description="Time window in days", ge=1, le=365)
    max_results: int = Field(200, description="Maximum results to return", ge=1, le=500)
    threshold: float = Field(0.4, description="Similarity threshold", ge=0.0, le=1.0)
    max_per_channel: Optional[int] = Field(None, description="Maximum results per channel (None = no limit)", ge=1, le=100)
    filter_speakers: Optional[List[str]] = Field(None, description="Filter by speaker hashes")
    filter_content_ids: Optional[List[int]] = Field(None, description="Filter by content IDs")
    filter_stitch_versions: Optional[List[str]] = Field(None, description="Filter by stitch versions (e.g., ['stitch_v14', 'stitch_v14.2'])")

    # Thematic analysis (NEW)
    generate_theme_summary: bool = Field(False, description="Generate thematic summary of results")
    extract_subthemes: bool = Field(False, description="Detect and extract sub-themes with adaptive validation")
    min_subtheme_silhouette_score: float = Field(0.15, description="Minimum silhouette score for valid sub-themes", ge=0.0, le=1.0)

    # Quantitative analysis (NEW)
    generate_quantitative_metrics: bool = Field(False, description="Generate quantitative metrics (segment counts, channel distribution, discourse centrality)")
    include_baseline_for_centrality: bool = Field(False, description="Include baseline comparison for discourse centrality calculation")


class QueryOptimizeRequest(BaseModel):
    """Request for query optimization"""
    query: str = Field(..., description="Original query to optimize")
    dashboard_id: str = Field("cprmv-practitioner", description="Dashboard ID for project filtering")
    user_email: Optional[str] = Field(None, description="User email for logging")


class RAGSummaryRequest(BaseModel):
    """Request for RAG summary generation"""
    query: str = Field(..., description="User query")
    segments: List[dict] = Field(..., description="List of segment dicts")
    dashboard_id: str = Field("cprmv-practitioner", description="Dashboard ID for project filtering")


class ThemeQueriesRequest(BaseModel):
    """Request for theme query generation"""
    theme_name: str = Field(..., description="Theme name")
    theme_description: str = Field(..., description="Theme description")
    dashboard_id: str = Field("cprmv-practitioner", description="Dashboard ID for project filtering")
    user_email: Optional[str] = Field(None, description="User email for logging")


class EmbeddingRequest(BaseModel):
    """Request for text embedding"""
    text: str = Field(..., description="Text to embed")
    model_type: Optional[str] = Field("default", description="Model type (default, alt)")


class GroupFilterRequest(BaseModel):
    """Filter configuration for a group"""
    channel_urls: Optional[List[str]] = Field(None, description="List of channel URLs")
    keywords: Optional[List[str]] = Field(None, description="Keywords to search in metadata")
    language: Optional[str] = Field(None, description="Language filter")
    projects: Optional[List[str]] = Field(None, description="Project filters")
    meta_data_query: Optional[dict] = Field(None, description="Custom metadata queries")
    start_date: Optional[datetime] = Field(None, description="Start date for time filtering")
    end_date: Optional[datetime] = Field(None, description="End date for time filtering")


class GroupConfigRequest(BaseModel):
    """Configuration for a single group"""
    group_id: str = Field(..., description="Unique group identifier")
    group_name: str = Field(..., description="Human-readable group name")
    filter: GroupFilterRequest = Field(..., description="Filter criteria")


class HierarchicalSummaryRequest(BaseModel):
    """Request for hierarchical summary generation"""
    # Time filtering (use either time_window_days OR start_date/end_date)
    time_window_days: Optional[int] = Field(None, description="Time window in days (convenient shorthand, defaults to 7 if no dates specified)", ge=1, le=365)
    start_date: Optional[datetime] = Field(None, description="Start date for time filtering (inclusive)")
    end_date: Optional[datetime] = Field(None, description="End date for time filtering (inclusive)")

    groupings: Optional[List[GroupConfigRequest]] = Field(None, description="Group configurations (optional - if None, analyzes all content)")
    theme_discovery_method: str = Field("clustering", description="Method: 'clustering' or 'predefined'")
    predefined_themes: Optional[List[dict]] = Field(None, description="Predefined themes if not using clustering")
    num_themes: Optional[int] = Field(None, description="Target number of themes to discover (None = auto)", ge=1, le=50)
    theme_selection_strategy: str = Field("aligned", description="Strategy: 'aligned' (discover themes across all groups together, weight equally) or 'top_per_group' (select top N themes per group independently)")
    clustering_params: Optional[dict] = Field(None, description="Clustering parameters (min_cluster_size, min_samples, etc.)")
    samples_per_theme: int = Field(20, description="Segments to sample per theme", ge=5, le=50)
    citation_format: str = Field("[G{group_id}-T{theme_id}-S{segment_id}]", description="Citation format string")
    generate_meta_summary: bool = Field(True, description="Generate cross-theme synthesis")
    synthesis_type: str = Field("cross_theme", description="Type of synthesis: cross_theme, cross_group, temporal")

    # Sub-theme extraction (NEW)
    extract_subthemes: bool = Field(False, description="Extract sub-themes within each theme using adaptive cluster validation")
    num_subthemes: Optional[int] = Field(3, description="Target number of sub-themes per theme (None = auto)", ge=2, le=10)
    subtheme_min_cluster_size: int = Field(3, description="Minimum cluster size for sub-themes", ge=2, le=10)
    require_valid_subtheme_clusters: bool = Field(True, description="Only extract sub-themes if clusters are well-separated")
    min_subtheme_silhouette_score: float = Field(0.15, description="Minimum silhouette score for valid sub-theme clusters", ge=0.0, le=1.0)

    # Quantitative analysis (NEW)
    generate_quantitative_metrics: bool = Field(False, description="Generate quantitative metrics (segment counts, channel distribution, discourse centrality)")
    include_baseline_for_centrality: bool = Field(False, description="Include baseline comparison for discourse centrality calculation")


class BatchEmbeddingRequest(BaseModel):
    """Request for batch text embedding"""
    texts: List[str] = Field(..., description="List of texts to embed")
    model_type: Optional[str] = Field("default", description="Model type (default, alt)")


class RetranscribeRequest(BaseModel):
    """Request for segment re-transcription"""
    segment_id: Optional[int] = Field(None, description="Segment ID to re-transcribe")
    content_id_int: Optional[int] = Field(None, description="Database content ID as integer (required if segment_id not provided)")
    content_id_str: Optional[str] = Field(None, description="YouTube/string content ID (alternative to content_id_int)")
    start_time: Optional[float] = Field(None, description="Start time in seconds (required if segment_id not provided)")
    end_time: Optional[float] = Field(None, description="End time in seconds (required if segment_id not provided)")
    provider: str = Field("assemblyai", description="Transcription provider (assemblyai, deepgram, etc.)")
    model: Optional[str] = Field("best", description="Model to use (best, nano, etc.)")
    language: Optional[str] = Field(None, description="Language code (auto-detect if None)")
    speaker_labels: bool = Field(True, description="Enable speaker diarization (Speaker A, Speaker B, etc.)")
    force: bool = Field(False, description="Force re-transcription even if cached")


class BatchRetranscribeRequest(BaseModel):
    """Request for batch re-transcription"""
    segment_ids: List[int] = Field(..., description="List of segment IDs to re-transcribe")
    provider: str = Field("assemblyai", description="Transcription provider")
    model: Optional[str] = Field("best", description="Model to use")
    language: Optional[str] = Field(None, description="Language code (auto-detect if None)")
    speaker_labels: bool = Field(True, description="Enable speaker diarization")
    force: bool = Field(False, description="Force re-transcription even if cached")


# ============================================================================
# Analysis API Models (Declarative Pipeline Configuration)
# ============================================================================

class StepConfig(BaseModel):
    """Configuration for a single pipeline step"""
    step: str = Field(..., description="Step name (e.g., 'expand_query', 'retrieve_segments')")
    config: dict = Field(default_factory=dict, description="Step-specific configuration parameters")


class AnalysisRequest(BaseModel):
    """
    Request for /api/analysis endpoint with declarative pipeline configuration.

    Two modes:
    1. Workflow shortcut: {"query": "...", "workflow": "simple_rag"}
    2. Custom pipeline: {"query": "...", "pipeline": [{...}, {...}]}
    """
    query: Optional[str] = Field(None, description="Analysis query text (required for query-based workflows, optional for discovery workflows like landing_page_overview)")
    dashboard_id: str = Field("cprmv-practitioner", description="Dashboard ID for project filtering")

    # Mode 1: Workflow shortcut (convenience)
    workflow: Optional[str] = Field(None, description="Predefined workflow name (e.g., 'simple_rag', 'hierarchical_summary')")

    # Mode 2: Custom pipeline (advanced)
    pipeline: Optional[List[StepConfig]] = Field(None, description="Ordered list of pipeline steps")

    # Global overrides (applied to all steps)
    config_overrides: Optional[dict] = Field(None, description="Override config for specific steps")

    # Filters (passed to all retrieval steps)
    time_window_days: Optional[int] = Field(None, description="Time window in days", ge=1, le=365)
    projects: Optional[List[str]] = Field(None, description="Project filters")
    languages: Optional[List[str]] = Field(None, description="Language filters")
    channels: Optional[List[str]] = Field(None, description="Channel filters")

    # Streaming options
    verbose: bool = Field(False, description="If True, emit progress and partial events. If False, only emit result and complete events.")

    # LLM backend selection
    use_local_llm: bool = Field(False, description="If True, use local LLM balancer instead of xAI API")


# ============================================================================
# Query API Models (Read-Only Segment Query)
# ============================================================================

class QueryFilters(BaseModel):
    """Filters for query API"""
    start_date: Optional[str] = Field(None, description="ISO date string (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="ISO date string (YYYY-MM-DD)")
    language: Optional[str] = Field(None, description="Language code (en, fr, etc.)")
    min_confidence: Optional[float] = Field(None, description="Minimum avg_confidence (0.0-1.0)", ge=0.0, le=1.0)
    content_ids: Optional[List[int]] = Field(None, description="Filter by content IDs")
    channel_names: Optional[List[str]] = Field(None, description="Filter by channel names")


class QuerySearch(BaseModel):
    """Search configuration for query API"""
    mode: str = Field(..., description="Search mode: 'semantic' or 'keyword'")
    query: str = Field(..., description="Search query text")


class QueryPagination(BaseModel):
    """Pagination configuration"""
    limit: int = Field(1000, description="Max results (1-5000)", ge=1, le=5000)
    offset: int = Field(0, description="Result offset", ge=0)


class QuerySort(BaseModel):
    """Sort configuration"""
    field: str = Field("publish_date", description="Sort field: publish_date, start_time, confidence")
    order: str = Field("desc", description="Sort order: asc or desc")


class QueryRequest(BaseModel):
    """Request for /api/query endpoint"""
    project: str = Field(..., description="Project ID (e.g., 'CPRMV')")
    filters: Optional[QueryFilters] = Field(None, description="Optional filters")
    search: Optional[QuerySearch] = Field(None, description="Optional search configuration")
    pagination: Optional[QueryPagination] = Field(None, description="Pagination config")
    sort: Optional[QuerySort] = Field(None, description="Sort config")
