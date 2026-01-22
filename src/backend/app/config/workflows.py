"""
Predefined Workflows & Caching System
=====================================

Convenience workflow definitions for common analysis patterns.

These can be referenced by name in AnalysisRequest:
    {"query": "...", "workflow": "simple_rag"}

Workflows are just predefined sequences of pipeline steps.


## Caching Architecture
=======================

The system uses a multi-level caching strategy to minimize latency:

### Cache Levels

1. **Query Variation Library** (Permanent)
   - Tables: `query_variations`, `query_expansions`
   - Stores: Unique query texts with their embeddings
   - Key insight: Same text = same embedding (never embed twice)
   - Variations can be reused across different original queries
   - Service: `QueryVariationService`

2. **Expand Query Cache** (via Query Variation Library)
   - Maps original query → expanded variations with embeddings
   - Time-agnostic (same expansion regardless of time window)
   - Service: `ExpandQueryCache`

3. **Workflow Result Cache** (Dynamic TTL)
   - Table: `llm_cache` (cache_type='workflow_result')
   - Stores: selected_segments, summaries, segment_ids
   - Cache key: query + time_window + projects + languages
   - TTL: 6h (7d queries), 12h (30d), 24h (longer)
   - Service: `WorkflowResultCache`

4. **Landing Page Cache** (24h TTL)
   - Table: `llm_cache` (cache_type='landing_page')
   - Stores: themes, corpus_stats, summaries
   - Cache key: time_window + projects + languages (no query)
   - Service: `LandingPageCache`


### Cache Flow for Query Workflows

```
Request arrives
    │
    ├─► Check WorkflowResultCache (exact match on query+filters)
    │   └─► HIT: Return cached result (instant for quick_summary)
    │
    ├─► Check ExpandQueryCache (query variations + embeddings)
    │   └─► HIT: Skip expand_query step, use cached embeddings
    │
    └─► MISS: Run full pipeline
            │
            ├─► expand_query: LLM generates variations (~2s)
            ├─► retrieve_segments: Embed queries + pgvector search (~6s)
            ├─► select_segments: Choose representative segments (~1s)
            └─► generate_summary: LLM summarization (~5s)
            │
            └─► Cache results for next time
```


### Performance Benchmarks

Tested with `quick_summary` workflow on live server (pre-warmed models):

| Scenario | Time | Notes |
|----------|------|-------|
| Cold (no cache) | ~14s | Full pipeline execution |
| Partial (expand cached) | ~10s | Skip LLM expansion, still embed+retrieve |
| Full cache hit | ~740ms | Instant return of cached result |

With ASGI transport (model loading overhead):
| Scenario | Time | Notes |
|----------|------|-------|
| Cold | ~49s | Includes model loading from disk |
| Warm | ~740ms | Same as live server |


### Workflow-Specific Cache Behavior

- **quick_summary**: Returns instantly on full cache hit (no fresh retrieval)
- **simple_rag**: Re-runs retrieval on cache hit for fresh quantitative stats
- **landing_page_overview**: 24h cache, returns instantly on hit


### Query Variation Library Design

The query variation library enables embedding reuse:

```sql
-- query_variations: Unique texts with embeddings
CREATE TABLE query_variations (
    id SERIAL PRIMARY KEY,
    text_hash VARCHAR(32) UNIQUE,  -- MD5 of normalized text
    text TEXT NOT NULL,
    embedding VECTOR(1024),         -- 0.6B model dimension
    usage_count INTEGER DEFAULT 1,
    created_at TIMESTAMP,
    last_used_at TIMESTAMP
);

-- query_expansions: Maps original queries to variations
CREATE TABLE query_expansions (
    id SERIAL PRIMARY KEY,
    original_query_hash VARCHAR(32),
    original_query TEXT,
    variation_id INTEGER REFERENCES query_variations(id),
    position INTEGER,
    UNIQUE(original_query_hash, variation_id)
);
```

Benefits:
- If "Mark Carney policy" appears in multiple query expansions, it's only embedded once
- Growing library of pre-embedded variations over time
- Deduplication via MD5 hash lookup (instant)


### Testing Cache Effectiveness

Use the test script to verify caching:
```bash
# Against live server (recommended)
BACKEND_URL=http://localhost:7999 uv run python src/backend/tests/test_quick_summary_workflow.py

# Against ASGI (includes model loading, slower)
uv run python src/backend/tests/test_quick_summary_workflow.py
```

The test runs the workflow twice to measure cache effectiveness.
"""

from typing import Dict, List, Any


# ============================================================================
# Workflow Definitions
# ============================================================================

WORKFLOWS: Dict[str, List[Dict[str, Any]]] = {
    "simple_rag": [
        {
            "step": "expand_query",
            "config": {
                "strategy": "multi_query"
            }
        },
        {
            "step": "retrieve_segments",
            "config": {
                "k": 200
            }
        },
        {
            "step": "rerank_segments",
            "config": {
                "best_per_episode": True,
                "similarity_weight": 0.40,
                "popularity_weight": 0.25,
                "recency_weight": 0.20,
                "single_speaker_weight": 0.08,
                "named_speaker_weight": 0.07,
                "similarity_floor": 0.55
            }
        },
        {
            "step": "quick_cluster_check",
            "config": {
                "method": "hdbscan",
                "min_cluster_size": 10,
                "min_silhouette_score": 0.25,
                "skip_if_few_segments": 30,
                "max_themes": 4,
                "force_split": True
            }
        },
        {
            "step": "quantitative_analysis",
            "config": {
                "include_baseline": True,
                "time_window_days": 7
            }
        },
        {
            "step": "select_segments",
            "config": {
                "strategy": "balanced",
                "n": 30,
                "n_unclustered": 10
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "rag_answer",
                "level": "theme",
                "model": "grok-4-fast-non-reasoning-latest",
                "temperature": 0.3,
                "max_tokens": 1200
            }
        }
    ],

    "search_only": [
        {
            "step": "expand_query",
            "config": {
                "strategy": "multi_query"
            }
        },
        {
            "step": "retrieve_segments",
            "config": {
                "k": 200
            }
        }
    ],

    "hierarchical_summary": [
        {
            "step": "retrieve_segments",
            "config": {}
        },
        {
            "step": "extract_themes",
            "config": {
                "method": "hdbscan",
                "min_cluster_size": 5,
                "min_samples": 3
            }
        },
        {
            "step": "quantitative_analysis",
            "config": {
                "include_baseline": False
            }
        },
        {
            "step": "select_segments",
            "config": {
                "strategy": "diversity",
                "n": 20
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "theme_summary",
                "level": "theme",
                "model": "grok-4-fast-non-reasoning-latest",
                "temperature": 0.3,
                "max_tokens": 400,
                "max_concurrent": 20
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "meta_summary",
                "level": "meta",
                "model": "grok-4-fast-non-reasoning-latest",
                "temperature": 0.3,
                "max_tokens": 800,
                "max_concurrent": 1
            }
        }
    ],

    "hierarchical_with_subthemes": [
        {
            "step": "retrieve_segments",
            "config": {}
        },
        {
            "step": "extract_themes",
            "config": {
                "method": "hdbscan",
                "min_cluster_size": 5,
                "min_samples": 3
            }
        },
        {
            "step": "extract_subthemes",
            "config": {
                "method": "hdbscan",
                "n_subthemes": 3,
                "min_cluster_size": 3,
                "require_valid_clusters": True,
                "min_silhouette_score": 0.15
            }
        },
        {
            "step": "quantitative_analysis",
            "config": {
                "include_baseline": False
            }
        },
        {
            "step": "select_segments",
            "config": {
                "strategy": "diversity",
                "n": 10
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "theme_summary",
                "level": "theme",
                "model": "grok-4-fast-non-reasoning-latest",
                "temperature": 0.3,
                "max_tokens": 400,
                "max_concurrent": 20
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "subtheme_summary",
                "level": "subtheme",
                "model": "grok-4-fast-non-reasoning-latest",
                "temperature": 0.3,
                "max_tokens": 300,
                "max_concurrent": 20
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "meta_summary",
                "level": "meta",
                "model": "grok-4-fast-non-reasoning-latest",
                "temperature": 0.3,
                "max_tokens": 800,
                "max_concurrent": 1
            }
        }
    ],

    "deep_analysis": [
        {
            "step": "expand_query",
            "config": {
                "strategy": "stance_variation",
                "n_stances": 5
            }
        },
        {
            "step": "retrieve_segments",
            "config": {
                "k": 300
            }
        },
        {
            "step": "extract_themes",
            "config": {
                "method": "hdbscan",
                "min_cluster_size": 5,
                "min_samples": 3
            }
        },
        {
            "step": "quantitative_analysis",
            "config": {
                "include_baseline": True
            }
        },
        {
            "step": "select_segments",
            "config": {
                "strategy": "diversity",
                "n": 20
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "theme_summary",
                "level": "theme",
                "model": "grok-4-fast-non-reasoning-latest",
                "temperature": 0.3,
                "max_tokens": 500,
                "max_concurrent": 20
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "meta_summary",
                "level": "meta",
                "model": "grok-4-fast-non-reasoning-latest",
                "temperature": 0.3,
                "max_tokens": 1000,
                "max_concurrent": 1
            }
        }
    ],

    "discourse_summary": [
        {
            "step": "retrieve_all_segments",
            "config": {
                "time_window_days": 14,
                "must_be_embedded": True,
                "must_be_stitched": True
            }
        },
        {
            "step": "corpus_analysis",
            "config": {
                "include_duration": True,
                "include_episode_count": True
            }
        },
        {
            "step": "extract_themes",
            "config": {
                "method": "hdbscan",
                "min_cluster_size": 50,
                "max_themes": 10,
                "min_theme_percentage": 4.0,
                "use_faiss": True
            }
        },
        {
            "step": "select_segments",
            "config": {
                "strategy": "diversity",
                "n": 15
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "theme_summary_with_metrics",
                "level": "theme",
                "max_concurrent": 10,
                "max_tokens": 500
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "discourse_portrait",
                "level": "corpus",
                "max_tokens": 1000,
                "include_all_theme_summaries": True
            }
        }
    ],

    # Local LLM variant of discourse_summary (uses tier_2 MLX model instead of Grok)
    "discourse_summary_local": [
        {
            "step": "retrieve_all_segments",
            "config": {
                "time_window_days": 14,
                "must_be_embedded": True,
                "must_be_stitched": True
            }
        },
        {
            "step": "corpus_analysis",
            "config": {
                "include_duration": True,
                "include_episode_count": True
            }
        },
        {
            "step": "extract_themes",
            "config": {
                "method": "hdbscan",
                "min_cluster_size": 50,
                "max_themes": 10,
                "min_theme_percentage": 4.0,
                "use_faiss": True
            }
        },
        {
            "step": "select_segments",
            "config": {
                "strategy": "diversity",
                "n": 15
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "theme_summary_with_metrics",
                "level": "theme",
                "max_concurrent": 10,
                "max_tokens": 500,
                "model": "tier_2",
                "backend": "local"
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "discourse_portrait",
                "level": "corpus",
                "max_tokens": 1000,
                "include_all_theme_summaries": True,
                "model": "tier_2",
                "backend": "local"
            }
        }
    ],

    # Alias for backward compatibility
    "landing_page_overview": [
        {
            "step": "retrieve_all_segments",
            "config": {
                "time_window_days": 14,
                "must_be_embedded": True,
                "must_be_stitched": True
            }
        },
        {
            "step": "corpus_analysis",
            "config": {
                "include_duration": True,
                "include_episode_count": True
            }
        },
        {
            "step": "extract_themes",
            "config": {
                "method": "hdbscan",
                "min_cluster_size": 50,
                "max_themes": 8,
                "min_theme_percentage": 5.0,
                "use_faiss": True
            }
        },
        {
            "step": "analyze_themes_with_subthemes",
            "config": {
                "quick_cluster_check": {
                    "min_cluster_size": 10,
                    "force_split": True,
                    "max_themes": 4
                },
                "quantitative_per_theme": True,
                "select_segments_per_subtheme": 8,
                "select_unclustered": 6,
                "generate_theme_names": True,
                "model": "grok-4-fast-non-reasoning-latest",
                "max_concurrent": 8
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "theme_summary_with_metrics",
                "level": "theme",
                "max_concurrent": 8,
                "max_tokens": 400
            }
        },
        {
            "step": "generate_summary",
            "config": {
                "template": "corpus_report_summary",
                "level": "corpus",
                "max_tokens": 800,
                "include_all_theme_summaries": True
            }
        }
    ],

    # ========================================================================
    # Quick Summary Workflow (Fast)
    # ========================================================================
    # Optimized for speed: expand query, but no clustering, fewer segments.
    # Target: 3-5 seconds for a complete summary.

    "quick_summary": [
        # Step 1: Expand query (still use multi_query for better recall)
        {
            "step": "expand_query",
            "config": {
                "strategy": "multi_query"
            }
        },
        # Step 2: Retrieve fewer segments for speed
        {
            "step": "retrieve_segments",
            "config": {
                "k": 50  # Fewer segments for speed
            }
        },
        # Step 3: Select top segments (no clustering, just balanced selection)
        {
            "step": "select_segments",
            "config": {
                "strategy": "balanced",
                "n": 15  # Fewer for faster LLM
            }
        },
        # Step 4: Generate summary with fast model
        {
            "step": "generate_summary",
            "config": {
                "template": "rag_answer",
                "level": "theme",
                "model": "grok-4-fast-non-reasoning-latest",
                "temperature": 0.3,
                "max_tokens": 600
            }
        }
    ],

}


# ============================================================================
# Workflow Functions
# ============================================================================

def get_workflow(name: str) -> List[Dict[str, Any]]:
    """
    Get workflow definition by name.

    Args:
        name: Workflow name

    Returns:
        List of step configs

    Raises:
        ValueError: If workflow not found
    """
    if name not in WORKFLOWS:
        raise ValueError(
            f"Unknown workflow: {name}. Available workflows: {list(WORKFLOWS.keys())}"
        )
    return WORKFLOWS[name]


def list_workflows() -> Dict[str, str]:
    """
    List all available workflows with descriptions.

    Returns:
        Dict mapping workflow name to description
    """
    return {
        "quick_summary": "Quick summary (2-3s): single embedding → retrieve → select top → fast summary (no clustering, no query expansion)",
        "simple_rag": "Simple RAG with adaptive clustering: expand query → retrieve → detect sub-themes (if meaningful) → analyze → sample per cluster → unified summary",
        "search_only": "Search only: expand query → retrieve (no summarization)",
        "hierarchical_summary": "Hierarchical: retrieve → cluster themes → summarize themes → meta-summary",
        "hierarchical_with_subthemes": "Hierarchical with sub-themes: retrieve → themes → sub-themes → summaries at all levels",
        "deep_analysis": "Deep analysis: stance variation → retrieve → themes → quantitative (with baseline) → summaries",
        "discourse_summary": "Discourse summary: retrieve all → corpus stats → cluster into 6-10 themes → select best segments → parallel theme summaries → overall discourse portrait",
        "discourse_summary_local": "Discourse summary (LOCAL LLM): Same as discourse_summary but uses local tier_2 MLX model (30B Qwen3) instead of Grok API",
        "landing_page_overview": "Landing page: retrieve all → discover major themes (FAISS + HDBSCAN) → corpus stats → parallel theme analysis with sub-themes → report-style summaries"
    }


def apply_config_overrides(
    workflow: List[Dict[str, Any]],
    overrides: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Apply config overrides to workflow steps.

    Args:
        workflow: Workflow definition (list of steps)
        overrides: Dict mapping step name to config overrides

    Returns:
        Modified workflow with overrides applied

    Example:
        workflow = get_workflow("simple_rag")
        overrides = {
            "expand_query": {"strategy": "theme_queries"},
            "retrieve_segments": {"k": 300}
        }
        workflow = apply_config_overrides(workflow, overrides)
    """
    result = []
    for step in workflow:
        step_copy = {**step}
        step_name = step_copy["step"]

        # Apply overrides if provided for this step
        if step_name in overrides:
            step_copy["config"] = {**step_copy["config"], **overrides[step_name]}

        result.append(step_copy)

    return result
