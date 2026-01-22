"""
Response Models
===============

Pydantic models for API responses.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class SearchResult(BaseModel):
    """Single search result"""
    segment_id: int
    similarity: float
    content_id: int
    content_id_string: Optional[str]
    channel_url: Optional[str]
    channel_name: Optional[str]
    title: Optional[str]
    publish_date: Optional[str]
    text: str
    start_time: float
    end_time: float
    speaker_hashes: List[str]
    segment_index: int
    stitch_version: Optional[str]


class SearchResponse(BaseModel):
    """Search API response"""
    success: bool = True
    results: List[SearchResult]
    total_results: int
    query: str
    time_window_days: int
    processing_time_ms: float

    # Thematic analysis (NEW)
    theme_summary: Optional[Dict[str, Any]] = None  # Overall theme summary
    subtheme_summaries: Optional[List[Dict[str, Any]]] = None  # Sub-theme summaries with validation

    # Quantitative metrics (NEW)
    quantitative_metrics: Optional[Dict[str, Any]] = None  # Quantitative analysis of results


class QueryOptimizeResponse(BaseModel):
    """Query optimization response"""
    success: bool = True
    keywords: List[str]
    query_variations: List[str]
    search_type: str
    processing_time_ms: float


class RAGSummaryResponse(BaseModel):
    """RAG summary response"""
    success: bool = True
    summary: Optional[str]
    segment_ids: List[int] = []  # IDs of segments used in summary
    keywords: List[str]
    segment_count: int
    volume_stats: Dict[str, Any]
    sampled_segments: List[dict]
    processing_time_ms: float


class ThemeQueriesResponse(BaseModel):
    """Theme queries response"""
    success: bool = True
    query_variations: List[str]
    search_type: str
    embeddings_1024: Optional[List[List[float]]]
    embeddings_2000: Optional[List[List[float]]]
    processing_time_ms: float


class EmbeddingResponse(BaseModel):
    """Embedding response"""
    success: bool = True
    embedding: List[float]
    dimension: int
    processing_time_ms: float


class BatchEmbeddingResponse(BaseModel):
    """Batch embedding response"""
    success: bool = True
    embeddings: List[List[float]]
    dimension: int
    count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    timestamp: float
    models_loaded: Optional[bool] = None
    db_connected: Optional[bool] = None


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None


class AlternativeTranscriptionDetail(BaseModel):
    """Single alternative transcription detail"""
    id: int
    provider: str
    model: Optional[str]
    language: Optional[str]
    transcription_text: str
    transcription_with_speakers: Optional[str] = Field(None, description="Formatted transcript with speaker labels (Speaker A: text)")
    translation_en: Optional[str] = Field(None, description="English translation (if source not English)")
    translation_fr: Optional[str] = Field(None, description="French translation (if source not French)")
    confidence: Optional[float]
    word_timings: Optional[List[Dict[str, Any]]]
    speaker_labels: Optional[List[Dict[str, Any]]]
    audio_duration: Optional[float]
    processing_time: Optional[float]
    created_at: str
    api_cost: Optional[float]


class TranscriptionResponse(BaseModel):
    """Response from re-transcription request"""
    success: bool = True
    segment_id: int
    content_id: int
    provider: str
    model: Optional[str]
    transcription: AlternativeTranscriptionDetail
    cached: bool
    processing_time_ms: float


class AlternativeTranscriptionsResponse(BaseModel):
    """Response listing all alternative transcriptions for a segment"""
    success: bool = True
    segment_id: int
    original_text: str
    alternatives: List[AlternativeTranscriptionDetail]
    total_count: int


class BatchTranscriptionResponse(BaseModel):
    """Response from batch re-transcription request"""
    success: bool = True
    total_requested: int
    successful: int
    failed: int
    results: List[TranscriptionResponse]
    errors: List[Dict[str, Any]]
    processing_time_ms: float


class HierarchicalSummaryResponse(BaseModel):
    """Response from hierarchical summary generation"""
    success: bool = True
    summary_id: str
    theme_summaries: List[Dict[str, Any]]
    subtheme_summaries: Optional[List[Dict[str, Any]]] = None  # Sub-themes with adaptive validation
    segment_ids_by_theme: Dict[str, List[int]] = {}  # theme_id -> segment IDs used
    group_summaries: Optional[Dict[str, Dict[str, Any]]] = None  # group_id -> MetaSummary dict
    cross_group_summary: Optional[Dict[str, Any]] = None  # Cross-group comparison
    meta_summary: Optional[Dict[str, Any]] = None  # Deprecated: use group_summaries/cross_group_summary
    total_themes: int
    total_subthemes: int = 0  # Number of sub-themes extracted
    total_groups: int
    total_citations: int
    processing_time_ms: float
    config_hash: str
    quantitative_metrics: Optional[Dict[str, Any]] = None  # NEW: Quantitative analysis metrics


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    details: Optional[str] = None


# ============================================================================
# Query API Response Models
# ============================================================================

class QuerySegmentResult(BaseModel):
    """Single segment result from query API"""
    segment_id: str = Field(..., description="Format: {content_id}_{floor(start_time)}")
    content_id: int
    content_id_string: Optional[str]
    title: Optional[str]
    channel_name: Optional[str]
    platform: Optional[str]
    publish_date: Optional[str]
    text: str
    start_time: float
    end_time: float
    duration: float
    language: Optional[str]
    confidence: Optional[float] = Field(None, description="From meta_data['avg_confidence']")
    speaker_hashes: List[str] = Field(default_factory=list)
    meta_data: Optional[Dict[str, Any]] = Field(None, description="Additional segment metadata")
    similarity: Optional[float] = Field(None, description="Similarity score (for semantic search)")


class QueryFiltersApplied(BaseModel):
    """Information about which filters were applied"""
    date_range: bool = False
    language: Optional[str] = None
    min_confidence: Optional[float] = None
    content_ids: bool = False
    channel_names: bool = False


class QueryInfo(BaseModel):
    """Metadata about the query execution"""
    project: str
    total_results: int
    returned_results: int
    filters_applied: QueryFiltersApplied
    search_mode: Optional[str] = None
    execution_time_ms: float


class QueryResponse(BaseModel):
    """Response from /api/query endpoint"""
    status: str = "success"
    query_info: QueryInfo
    results: List[QuerySegmentResult]


# ============================================================================
# Entity Detail Response Models
# ============================================================================

class SpeakerPreview(BaseModel):
    """Brief speaker info for embedding in other responses"""
    speaker_id: str  # speaker_hash or speaker_identity_id
    name: str
    appearance_count: Optional[int] = None
    image_url: Optional[str] = None


class EpisodePreview(BaseModel):
    """Brief episode info for embedding in other responses"""
    content_id: str
    content_id_numeric: int
    title: str
    channel_name: str
    channel_id: Optional[int] = None
    publish_date: Optional[str] = None
    duration: Optional[int] = None  # seconds
    platform: Optional[str] = None


class ChannelPreview(BaseModel):
    """Brief channel info for embedding in other responses"""
    channel_id: int
    channel_key: str
    name: str
    platform: Optional[str] = None
    episode_count: Optional[int] = None
    image_url: Optional[str] = None


class DateRange(BaseModel):
    """Date range for content"""
    earliest: str
    latest: str


class ChannelWithCount(BaseModel):
    """Channel with appearance count"""
    channel_id: int
    channel_key: str
    name: str
    platform: Optional[str] = None
    count: int


class TopicWithCount(BaseModel):
    """Topic with occurrence count"""
    topic: str
    count: int


class EpisodeDetailsResponse(BaseModel):
    """Full episode details response"""
    success: bool = True
    content_id: int
    content_id_string: str
    title: str
    description: Optional[str] = None
    channel_id: Optional[int] = None
    channel_name: str
    channel_key: Optional[str] = None
    platform: str
    publish_date: Optional[str] = None
    duration: Optional[int] = None  # seconds
    main_language: Optional[str] = None

    # Processing state
    has_transcript: bool = False
    has_diarization: bool = False
    has_embeddings: bool = False

    # Metadata
    source_url: Optional[str] = None
    thumbnail_url: Optional[str] = None

    # Speakers
    speakers: List[SpeakerPreview] = Field(default_factory=list)

    # Stats
    segment_count: int = 0

    # Related episodes (same channel)
    related_episodes: List[EpisodePreview] = Field(default_factory=list)

    processing_time_ms: float


class SpeakerDetailsResponse(BaseModel):
    """Full speaker details response"""
    success: bool = True
    speaker_id: str  # speaker_hash or identity ID
    name: str

    # From SpeakerIdentity if available
    bio: Optional[str] = None
    occupation: Optional[str] = None
    organization: Optional[str] = None
    role: Optional[str] = None
    image_url: Optional[str] = None

    # Aggregated stats
    total_appearances: int = 0
    total_duration_seconds: float = 0.0
    total_episodes: int = 0

    # Recent episodes
    recent_episodes: List[EpisodePreview] = Field(default_factory=list)

    # Top channels by appearance
    top_channels: List[ChannelWithCount] = Field(default_factory=list)

    # Related speakers (co-appear frequently)
    related_speakers: List[SpeakerPreview] = Field(default_factory=list)

    processing_time_ms: float


class ChartPosition(BaseModel):
    """A single chart position"""
    platform: str  # e.g., "apple"
    country: str  # e.g., "ca", "us"
    category: str  # e.g., "all-podcasts", "news"
    rank: int  # Position on chart (1-200)


class ChannelRanking(BaseModel):
    """Channel importance/ranking information"""
    importance_score: Optional[float] = None  # Raw score from database
    importance_rank: Optional[int] = None  # Rank among all channels (1 = highest)
    tier: Optional[str] = None  # "top-10", "top-50", "top-100", "top-200", or None
    chart_positions: List[ChartPosition] = Field(default_factory=list)  # Recent chart positions
    chart_month: Optional[str] = None  # Month of chart data (e.g., "2026-01")


class ChannelDetailsResponse(BaseModel):
    """Full channel details response"""
    success: bool = True
    channel_id: int
    channel_key: str
    name: str
    platform: str
    description: Optional[str] = None
    primary_url: Optional[str] = None
    language: Optional[str] = None
    status: Optional[str] = None
    image_url: Optional[str] = None

    # Stats
    episode_count: int = 0
    date_range: Optional[DateRange] = None
    publishing_frequency: Optional[str] = None

    # Importance/Ranking
    ranking: Optional[ChannelRanking] = None

    # Regular speakers
    regular_speakers: List[SpeakerPreview] = Field(default_factory=list)

    # Recent episodes
    recent_episodes: List[EpisodePreview] = Field(default_factory=list)

    # Tags
    tags: List[str] = Field(default_factory=list)

    processing_time_ms: float
