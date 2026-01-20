"""
Step Registry
=============

Maps step names to AnalysisPipeline methods and provides metadata.

This enables the /api/analysis endpoint to build pipelines dynamically
from declarative JSON configurations.
"""

from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass


@dataclass
class StepMetadata:
    """Metadata for a pipeline step"""
    name: str
    description: str
    parameters: Dict[str, Any]
    method_name: str  # Method name on AnalysisPipeline


# Step registry: name → metadata
STEP_REGISTRY: Dict[str, StepMetadata] = {
    "expand_query": StepMetadata(
        name="expand_query",
        description="Expand query into multiple variations for improved recall",
        parameters={
            "strategy": {
                "type": "string",
                "enum": ["multi_query", "query2doc", "theme_queries", "stance_variation"],
                "default": "multi_query",
                "description": "Query expansion strategy"
            },
            "n_stances": {
                "type": "integer",
                "default": 3,
                "description": "Number of stances (for stance_variation strategy)"
            }
        },
        method_name="expand_query"
    ),

    "retrieve_segments": StepMetadata(
        name="retrieve_segments",
        description="Retrieve segments using semantic search with optional keyword filtering (hybrid search)",
        parameters={
            "k": {
                "type": "integer",
                "default": 200,
                "description": "Maximum results per query embedding"
            },
            "threshold": {
                "type": "number",
                "default": 0.42,
                "description": "Minimum similarity threshold (0.0-1.0). Minimum 0.42 recommended."
            },
            "time_window_days": {
                "type": "integer",
                "default": 30,
                "description": "Time window in days"
            },
            "must_contain": {
                "type": "array",
                "default": None,
                "description": "Keywords that ALL must appear in text (AND). For entity queries like 'Mark Carney'."
            },
            "must_contain_any": {
                "type": "array",
                "default": None,
                "description": "Keywords where AT LEAST ONE must appear (OR). For name variants like ['Carney', 'Mark Carney']."
            },
            "projects": {
                "type": "array",
                "default": None,
                "description": "Project filters"
            },
            "languages": {
                "type": "array",
                "default": None,
                "description": "Language filters"
            },
            "channels": {
                "type": "array",
                "default": None,
                "description": "Channel filters"
            }
        },
        method_name="retrieve_segments_by_search"
    ),

    "extract_themes": StepMetadata(
        name="extract_themes",
        description="Extract themes from segments using HDBSCAN clustering",
        parameters={
            "method": {
                "type": "string",
                "enum": ["hdbscan", "kmeans"],
                "default": "hdbscan",
                "description": "Clustering method"
            },
            "max_themes": {
                "type": "integer",
                "default": 10,
                "description": "Maximum number of themes to extract"
            },
            "min_theme_percentage": {
                "type": "number",
                "default": 5.0,
                "description": "Minimum percentage of segments for a valid theme"
            },
            "use_faiss": {
                "type": "boolean",
                "default": False,
                "description": "Use FAISS for fast similarity search"
            },
            "n_clusters": {
                "type": "integer",
                "default": None,
                "description": "Number of clusters (None = auto)"
            },
            "min_cluster_size": {
                "type": "integer",
                "default": 5,
                "description": "Minimum cluster size"
            },
            "min_samples": {
                "type": "integer",
                "default": 3,
                "description": "Minimum samples for core point"
            }
        },
        method_name="extract_themes"
    ),

    "quick_cluster_check": StepMetadata(
        name="quick_cluster_check",
        description="Fast clustering validation for sub-theme detection with adaptive validation (only creates clusters if meaningful)",
        parameters={
            "method": {
                "type": "string",
                "enum": ["hdbscan", "fast"],
                "default": "fast",
                "description": "Clustering method: 'fast' (PCA+KMeans, ~50ms) or 'hdbscan' (UMAP+HDBSCAN, ~1-5s)"
            },
            "min_cluster_size": {
                "type": "integer",
                "default": 10,
                "description": "Minimum segments per cluster"
            },
            "min_silhouette_score": {
                "type": "float",
                "default": 0.15,
                "description": "Minimum silhouette score for valid clusters (0.15 = weak but detectable structure)"
            },
            "skip_if_few_segments": {
                "type": "integer",
                "default": 30,
                "description": "Skip clustering if fewer than this many segments"
            },
            "max_themes": {
                "type": "integer",
                "default": 4,
                "description": "Maximum number of clusters to keep"
            },
            "force_split": {
                "type": "boolean",
                "default": True,
                "description": "Try k-means fallback (k=2,3,4) when HDBSCAN finds 0-1 clusters"
            }
        },
        method_name="quick_cluster_check"
    ),

    "extract_subthemes": StepMetadata(
        name="extract_subthemes",
        description="Extract sub-themes within each theme using clustering",
        parameters={
            "method": {
                "type": "string",
                "enum": ["hdbscan"],
                "default": "hdbscan",
                "description": "Clustering method"
            },
            "n_subthemes": {
                "type": "integer",
                "default": 3,
                "description": "Target number of sub-themes per theme"
            },
            "min_cluster_size": {
                "type": "integer",
                "default": 3,
                "description": "Minimum cluster size"
            },
            "require_valid_clusters": {
                "type": "boolean",
                "default": True,
                "description": "Only extract sub-themes if clusters are well-separated"
            },
            "min_silhouette_score": {
                "type": "float",
                "default": 0.15,
                "description": "Minimum silhouette score for valid clusters"
            }
        },
        method_name="extract_subthemes"
    ),

    "quantitative_analysis": StepMetadata(
        name="quantitative_analysis",
        description="Generate quantitative metrics: segment counts, channel/video distribution, discourse centrality",
        parameters={
            "include_baseline": {
                "type": "boolean",
                "default": False,
                "description": "Include baseline comparison for discourse centrality"
            },
            "time_window_days": {
                "type": "integer",
                "default": 7,
                "description": "Time window for baseline (if include_baseline=True)"
            }
        },
        method_name="quantitative_analysis"
    ),

    "rerank_segments": StepMetadata(
        name="rerank_segments",
        description="Rerank retrieved segments by popularity, recency, speaker quality; apply diversity constraints",
        parameters={
            "best_per_episode": {
                "type": "boolean",
                "default": True,
                "description": "Keep only best segment per episode (content_id)"
            },
            "max_per_channel": {
                "type": "integer",
                "default": None,
                "description": "Optional max segments per channel"
            },
            "similarity_weight": {
                "type": "number",
                "default": 0.4,
                "description": "Weight for semantic similarity score"
            },
            "popularity_weight": {
                "type": "number",
                "default": 0.2,
                "description": "Weight for channel popularity (importance_score)"
            },
            "recency_weight": {
                "type": "number",
                "default": 0.2,
                "description": "Weight for recency (newer = higher)"
            },
            "single_speaker_weight": {
                "type": "number",
                "default": 0.1,
                "description": "Bonus weight for single-speaker segments"
            },
            "named_speaker_weight": {
                "type": "number",
                "default": 0.1,
                "description": "Bonus weight for segments with named speakers"
            },
            "time_window_days": {
                "type": "integer",
                "default": 30,
                "description": "Time window for recency normalization"
            }
        },
        method_name="rerank_segments"
    ),

    "select_segments": StepMetadata(
        name="select_segments",
        description="Select diverse subset of segments from each theme",
        parameters={
            "strategy": {
                "type": "string",
                "enum": ["diversity", "balanced", "recency"],
                "default": "balanced",
                "description": "Selection strategy"
            },
            "n": {
                "type": "integer",
                "default": 20,
                "description": "Number of segments to select per theme"
            },
            "n_unclustered": {
                "type": "integer",
                "default": 6,
                "description": "Number of unclustered/outlier segments to include (when using clustering)"
            }
        },
        method_name="select_segments"
    ),

    "generate_summary": StepMetadata(
        name="generate_summary",
        description="Generate LLM summary with citations",
        parameters={
            "template": {
                "type": "string",
                "default": "rag_answer",
                "description": "Prompt template name"
            },
            "level": {
                "type": "string",
                "enum": ["theme", "subtheme", "meta", "domain", "corpus"],
                "default": "theme",
                "description": "Summary level"
            },
            "model": {
                "type": "string",
                "default": "grok-4-fast-non-reasoning-latest",
                "description": "LLM model to use. For xAI: 'grok-2-1212', 'grok-4-fast-non-reasoning-latest'. For local: 'tier_1' (80B), 'tier_2' (30B), 'tier_3' (4B)"
            },
            "backend": {
                "type": "string",
                "enum": ["xai", "local"],
                "default": "xai",
                "description": "LLM backend: 'xai' for Grok API, 'local' for local MLX model servers"
            },
            "temperature": {
                "type": "float",
                "default": 0.3,
                "description": "Generation temperature"
            },
            "max_tokens": {
                "type": "integer",
                "default": 400,
                "description": "Maximum tokens"
            },
            "max_concurrent": {
                "type": "integer",
                "default": 20,
                "description": "Maximum concurrent generation tasks"
            },
            "include_subtheme_summaries": {
                "type": "boolean",
                "default": False,
                "description": "Include sub-theme summaries in theme-level generation"
            },
            "include_theme_summaries": {
                "type": "boolean",
                "default": False,
                "description": "Include theme summaries in domain-level generation"
            },
            "include_all_theme_summaries": {
                "type": "boolean",
                "default": False,
                "description": "Include all theme summaries in corpus-level generation"
            },
            "include_all_domain_summaries": {
                "type": "boolean",
                "default": False,
                "description": "Include all domain summaries in corpus-level generation"
            }
        },
        method_name="generate_summaries"
    ),

    "group_by": StepMetadata(
        name="group_by",
        description="Group segments by field (e.g., channel, language)",
        parameters={
            "field": {
                "type": "string",
                "description": "Field to group by (e.g., 'language', 'channel_url')"
            }
        },
        method_name="group_by"
    ),

    "retrieve_all_segments": StepMetadata(
        name="retrieve_all_segments",
        description="Retrieve ALL segments matching filters (no search/query). For landing page workflows.",
        parameters={
            "time_window_days": {
                "type": "integer",
                "default": 30,
                "description": "Time window in days"
            },
            "projects": {
                "type": "array",
                "default": None,
                "description": "Project filters"
            },
            "languages": {
                "type": "array",
                "default": None,
                "description": "Language filters"
            },
            "channels": {
                "type": "array",
                "default": None,
                "description": "Channel filters"
            },
            "must_be_stitched": {
                "type": "boolean",
                "default": True,
                "description": "Only stitched segments"
            },
            "must_be_embedded": {
                "type": "boolean",
                "default": True,
                "description": "Only embedded segments"
            }
        },
        method_name="retrieve_all_segments"
    ),

    "corpus_analysis": StepMetadata(
        name="corpus_analysis",
        description="Generate corpus-level quantitative analysis (episodes, duration, channels, etc.)",
        parameters={
            "include_duration": {
                "type": "boolean",
                "default": True,
                "description": "Calculate total duration in hours"
            },
            "include_episode_count": {
                "type": "boolean",
                "default": True,
                "description": "Count unique episodes"
            }
        },
        method_name="corpus_analysis"
    ),

    "analyze_themes_with_subthemes": StepMetadata(
        name="analyze_themes_with_subthemes",
        description="Analyze each theme with sub-themes in parallel (quick_cluster_check, quantitative, LLM naming, selection)",
        parameters={
            "quick_cluster_check": {
                "type": "object",
                "default": {},
                "description": "Config for quick_cluster_check step"
            },
            "quantitative_per_theme": {
                "type": "boolean",
                "default": True,
                "description": "Run quantitative analysis per theme"
            },
            "select_segments_per_subtheme": {
                "type": "integer",
                "default": 8,
                "description": "Segments to select per sub-theme"
            },
            "select_unclustered": {
                "type": "integer",
                "default": 6,
                "description": "Unclustered segments to include"
            },
            "generate_theme_names": {
                "type": "boolean",
                "default": True,
                "description": "Generate LLM names for themes"
            },
            "model": {
                "type": "string",
                "default": "grok-4-fast-non-reasoning-latest",
                "description": "LLM model for theme naming"
            },
            "max_concurrent": {
                "type": "integer",
                "default": 8,
                "description": "Max concurrent theme analysis tasks"
            }
        },
        method_name="analyze_themes_with_subthemes"
    ),

}


def get_step_metadata(step_name: str) -> StepMetadata:
    """
    Get metadata for a step.

    Args:
        step_name: Step name (e.g., 'expand_query')

    Returns:
        StepMetadata

    Raises:
        ValueError: If step not found
    """
    if step_name not in STEP_REGISTRY:
        raise ValueError(
            f"Unknown step: {step_name}. Available steps: {list(STEP_REGISTRY.keys())}"
        )
    return STEP_REGISTRY[step_name]


def list_steps() -> List[Dict[str, Any]]:
    """
    List all available steps with metadata.

    Returns:
        List of step metadata dicts
    """
    return [
        {
            "name": meta.name,
            "description": meta.description,
            "parameters": meta.parameters
        }
        for meta in STEP_REGISTRY.values()
    ]


def validate_step_config(step_name: str, config: Dict[str, Any]) -> None:
    """
    Validate step configuration.

    Args:
        step_name: Step name
        config: Configuration dict

    Raises:
        ValueError: If configuration invalid
    """
    metadata = get_step_metadata(step_name)

    # Check for unknown parameters
    valid_params = set(metadata.parameters.keys())
    provided_params = set(config.keys())
    unknown = provided_params - valid_params

    if unknown:
        raise ValueError(
            f"Unknown parameters for step '{step_name}': {unknown}. "
            f"Valid parameters: {valid_params}"
        )


def build_pipeline_from_steps(
    pipeline,
    query: Optional[str],
    steps: List[Dict[str, Any]],
    global_filters: Dict[str, Any] = None
):
    """
    Build AnalysisPipeline from step definitions.

    Args:
        pipeline: AnalysisPipeline instance
        query: Query text
        steps: List of step configs ({"step": "name", "config": {...}})
        global_filters: Global filters to merge into all steps (projects, languages, etc.)

    Returns:
        AnalysisPipeline with steps added

    Example:
        steps = [
            {"step": "expand_query", "config": {"strategy": "multi_query"}},
            {"step": "retrieve_segments", "config": {"k": 200}},
            {"step": "select_segments", "config": {"n": 20}},
            {"step": "generate_summary", "config": {}}
        ]
        pipeline = build_pipeline_from_steps(pipeline, "climate change", steps)
    """
    global_filters = global_filters or {}

    for step_def in steps:
        step_name = step_def["step"]
        config = step_def.get("config", {})

        # Validate
        validate_step_config(step_name, config)

        # Get metadata
        metadata = get_step_metadata(step_name)
        method_name = metadata.method_name

        # Merge global filters for retrieval and analysis steps
        # Global filters should override workflow defaults (so config goes first, then global_filters override)
        if step_name in ["retrieve_segments", "quantitative_analysis", "retrieve_segments_by_search",
                         "retrieve_all_segments", "rerank_segments"]:
            config = {**config, **global_filters}

        # Pass use_local_llm to LLM generation steps (summaries and query expansion)
        if step_name in ["generate_summary", "generate_summaries", "expand_query"] and global_filters.get("use_local_llm"):
            config = {**config, "backend": "local"}

        # Get method from pipeline
        method = getattr(pipeline, method_name)

        # Special handling for steps that need query as first arg
        if step_name == "expand_query":
            pipeline = method(query, **config)
        else:
            pipeline = method(**config)

    return pipeline
