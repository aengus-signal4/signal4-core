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
        description="Retrieve segments using semantic search with pgvector",
        parameters={
            "k": {
                "type": "integer",
                "default": 200,
                "description": "Maximum results per query embedding"
            },
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
                "enum": ["hdbscan"],
                "default": "hdbscan",
                "description": "Clustering method"
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
                "description": "LLM model to use"
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

    # ========================================================================
    # Health Dashboard Steps
    # ========================================================================

    "generate_health_queries": StepMetadata(
        name="generate_health_queries",
        description="Generate diverse queries across health domains (nutrition, fitness, mental health, etc.)",
        parameters={
            "include_llm_expansion": {
                "type": "boolean",
                "default": True,
                "description": "Expand queries with LLM variations"
            },
            "include_cross_domain": {
                "type": "boolean",
                "default": True,
                "description": "Include cross-domain connection queries"
            },
            "include_perspectives": {
                "type": "boolean",
                "default": True,
                "description": "Include perspective variation queries (mainstream, alternative, skeptical)"
            },
            "max_queries": {
                "type": "integer",
                "default": 100,
                "description": "Maximum queries to generate"
            },
            "sampling_strategy": {
                "type": "string",
                "enum": ["weighted", "stratified", "random"],
                "default": "stratified",
                "description": "Query sampling strategy"
            }
        },
        method_name="generate_health_queries"
    ),

    "batch_retrieve_segments": StepMetadata(
        name="batch_retrieve_segments",
        description="Retrieve segments for multiple queries in batch with deduplication",
        parameters={
            "k_per_query": {
                "type": "integer",
                "default": 50,
                "description": "Maximum results per query"
            },
            "time_window_days": {
                "type": "integer",
                "default": 90,
                "description": "Time window in days"
            },
            "deduplicate": {
                "type": "boolean",
                "default": True,
                "description": "Deduplicate segments across queries"
            },
            "max_total_segments": {
                "type": "integer",
                "default": 2000,
                "description": "Maximum total segments to return"
            }
        },
        method_name="batch_retrieve_segments"
    ),

    "group_by_domain": StepMetadata(
        name="group_by_domain",
        description="Group segments by health domain using query metadata",
        parameters={
            "min_segments_per_domain": {
                "type": "integer",
                "default": 10,
                "description": "Minimum segments required to include a domain"
            },
            "domains_from": {
                "type": "string",
                "enum": ["query_metadata", "embedding_clusters", "llm_classification"],
                "default": "query_metadata",
                "description": "How to determine segment domains"
            }
        },
        method_name="group_by_domain"
    )
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
                         "batch_retrieve_segments", "retrieve_all_segments"]:
            config = {**config, **global_filters}

        # Get method from pipeline
        method = getattr(pipeline, method_name)

        # Special handling for steps that need query as first arg
        if step_name == "expand_query":
            pipeline = method(query, **config)
        else:
            pipeline = method(**config)

    return pipeline
