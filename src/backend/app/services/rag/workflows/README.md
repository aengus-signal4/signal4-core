# RAG Workflow System

Composable analysis pipeline framework for building RAG (Retrieval-Augmented Generation) workflows.

## Architecture Overview

The workflow system uses a **pipeline-based architecture** with three layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    Workflow Layer                           │
│   SimpleRAGWorkflow, HierarchicalSummaryWorkflow            │
│   Pre-built workflows for common analysis patterns          │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Layer                           │
│   AnalysisPipeline                                          │
│   Fluent API for composing steps, streaming execution       │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   Component Layer                           │
│   SegmentRetriever, ThemeExtractor, SegmentSelector,        │
│   TextGenerator, QuantitativeAnalyzer                       │
│   Reusable components with Protocol interfaces              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Using Pre-built Workflows

```python
from services.rag.workflows import SimpleRAGWorkflow, HierarchicalSummaryWorkflow

# Simple Q&A with query expansion
workflow = SimpleRAGWorkflow(llm_service, db_session)
result = await workflow.run_with_expansion(
    query="What is being said about carbon taxes?",
    expansion_strategy="multi_query",
    k=100,
    n_samples=20,
    projects=["Canada"],
    languages=["en"]
)
print(result["summary"])

# Hierarchical multi-group analysis
workflow = HierarchicalSummaryWorkflow(llm_service, db_session)
result = await workflow.run(
    groupings=[
        {"group_id": "en", "group_name": "English", "filter": {"languages": ["en"]}},
        {"group_id": "fr", "group_name": "French", "filter": {"languages": ["fr"]}}
    ],
    num_themes=5,
    time_window_days=30
)
```

### Building Custom Pipelines

```python
from services.rag.analysis_pipeline import AnalysisPipeline

pipeline = AnalysisPipeline(
    "custom_analysis",
    llm_service=llm_service,
    db_session=session
)

# Build pipeline with fluent API
pipeline = (
    pipeline
    .expand_query("climate policy debate", strategy="theme_queries")
    .retrieve_segments_by_search(k=200, threshold=0.45)
    .rerank_segments(best_per_episode=True)
    .extract_themes(method="hdbscan", max_themes=10)
    .select_segments(strategy="diversity", n=20)
    .generate_summaries(template="theme_summary", level="theme")
)

# Execute with streaming
async for event in pipeline.execute():
    if event["type"] == "step_complete":
        print(f"Completed: {event['step']}")
    elif event["type"] == "complete":
        results = event["data"]
```

## Workflow Classes

### SimpleRAGWorkflow

Direct Q&A workflow with two modes:

**`run()`** - Use pre-retrieved segments:
```python
result = await workflow.run(
    query="What is the debate about?",
    segments=search_results,  # Pre-retrieved segments
    n_samples=20,
    model="grok-4-fast-non-reasoning-latest"
)
```

**`run_with_expansion()`** - Full pipeline with query expansion:
```python
result = await workflow.run_with_expansion(
    query="Pierre Poilievre economic policy",
    expansion_strategy="multi_query",  # or "query2doc", "stance_variation"
    k=100,
    n_samples=20,
    time_window_days=30,
    projects=["Canada"],
    generate_quantitative_metrics=True
)
```

**`run_with_expansion_stream()`** - Streaming version with progress events:
```python
async for event in workflow.run_with_expansion_stream(query="..."):
    print(event["type"], event.get("step"))
```

### HierarchicalSummaryWorkflow

Multi-group, multi-theme analysis with hierarchical summarization:

```python
result = await workflow.run(
    groupings=[
        {"group_id": "en", "group_name": "English", "filter": {"languages": ["en"]}},
        {"group_id": "fr", "group_name": "French", "filter": {"languages": ["fr"]}}
    ],
    time_window_days=30,
    num_themes=5,
    samples_per_theme=20,
    extract_subthemes=True,  # Enable sub-theme extraction
    n_subthemes=3
)
```

## Pipeline Steps

All steps are registered in `step_registry.py` and can be used via the fluent API.

### Query Expansion

| Step | Description |
|------|-------------|
| `expand_query(query, strategy)` | Expand query into multiple variations |

**Strategies:**
- `multi_query`: 10 diverse variations (5 EN + 5 FR)
- `query2doc`: Generate pseudo-document
- `theme_queries`: Discourse-focused variations
- `stance_variation`: Multiple stance perspectives

### Retrieval

| Step | Description |
|------|-------------|
| `retrieve_segments_by_search(k, threshold, ...)` | Semantic search with filters |
| `retrieve_all_segments(time_window_days, ...)` | Fetch all segments (no search) |

**Common parameters:**
- `k`: Max results per query embedding (default: 200)
- `threshold`: Similarity threshold 0-1 (default: 0.42)
- `time_window_days`: Time filter in days
- `must_contain`: Keywords that ALL must appear (AND)
- `must_contain_any`: Keywords where at least one must appear (OR)
- `projects`, `languages`, `channels`: Filter lists

### Reranking

| Step | Description |
|------|-------------|
| `rerank_segments(...)` | Rerank by popularity, recency, speaker quality |

**Parameters:**
- `best_per_episode`: Keep only best segment per episode (default: True)
- `max_per_channel`: Optional max per channel
- `similarity_weight`, `popularity_weight`, `recency_weight`: Score weights

### Theme Extraction

| Step | Description |
|------|-------------|
| `extract_themes(method, max_themes, ...)` | Cluster segments into themes |
| `quick_cluster_check(...)` | Fast clustering validation |
| `extract_subthemes(n_subthemes, ...)` | Extract sub-themes within themes |

**Parameters:**
- `method`: "hdbscan" (default) or "kmeans"
- `max_themes`: Maximum themes to extract
- `min_cluster_size`: Minimum segments per cluster
- `min_theme_percentage`: Minimum % of segments for valid theme

### Selection

| Step | Description |
|------|-------------|
| `select_segments(strategy, n)` | Select diverse subset per theme |

**Strategies:**
- `diversity`: Maximize diversity (MMR-style)
- `balanced`: Balance diversity and centrality
- `recency`: Prefer recent segments

### Analysis

| Step | Description |
|------|-------------|
| `quantitative_analysis(include_baseline)` | Generate quantitative metrics |
| `corpus_analysis(include_duration)` | Corpus-level statistics |
| `analyze_themes_with_subthemes(...)` | Full theme analysis with sub-themes |

### Generation

| Step | Description |
|------|-------------|
| `generate_summaries(template, level, ...)` | Generate LLM summaries |

**Parameters:**
- `template`: Prompt template name ("theme_summary", "rag_answer", etc.)
- `level`: "theme", "subtheme", "meta", "domain", "corpus"
- `model`: LLM model to use
- `max_concurrent`: Parallel generation limit

### Grouping

| Step | Description |
|------|-------------|
| `group_by(field)` | Group segments by field (channel, language) |

## Result Formats

### WorkflowEvent (streaming)

```python
{
    "type": "step_start" | "step_progress" | "step_complete" | "result" | "complete" | "error",
    "step": "expand_query",
    "progress": 5,
    "total": 20,
    "data": {...}
}
```

### SimpleRAGResult

```python
{
    "summary": "Generated answer with citations...",
    "segment_ids": [123, 456, 789],
    "segment_count": 100,
    "samples_used": 20,
    "expanded_queries": ["query1", "query2", ...],
    "expansion_strategy": "multi_query",
    "search_results": [...],  # Full segment metadata
    "quantitative_metrics": {...}  # Optional
}
```

### HierarchicalResult

```python
{
    "theme_summaries": [
        {
            "group_id": "en",
            "theme_id": "en_theme_0",
            "theme_name": "Economic Policy",
            "summary": "...",
            "segment_count": 45,
            "keywords": ["economy", "tax", ...]
        },
        ...
    ],
    "segment_ids_by_theme": {"en_theme_0": [123, 456], ...},
    "group_results": {...},
    "total_themes": 10,
    "total_segments": 500,
    "quantitative_metrics": {...}
}
```

## Protocol Interfaces

The system defines Protocol interfaces (PEP 544) for all components, enabling structural subtyping and easy testing.

### Importing Protocols

```python
from services.rag.workflows import (
    # Workflow protocols
    WorkflowProtocol,
    StreamingWorkflowProtocol,
    # Component protocols
    SegmentRetrieverProtocol,
    ThemeExtractorProtocol,
    SegmentSelectorProtocol,
    TextGeneratorProtocol,
    QuantitativeAnalyzerProtocol,
    # TypedDicts
    WorkflowEvent,
    SimpleRAGResult,
    HierarchicalResult,
)
```

### Using Protocols for Type Hints

```python
def execute_workflow(workflow: WorkflowProtocol, **kwargs) -> dict:
    return await workflow.run(**kwargs)

def process_events(events: AsyncGenerator[WorkflowEvent, None]):
    async for event in events:
        if event["type"] == "complete":
            return event["data"]
```

### Creating Custom Components

Implement the protocol to create swappable components:

```python
class CustomThemeExtractor:
    """Custom theme extractor that conforms to ThemeExtractorProtocol."""

    def extract_by_clustering(
        self,
        segments: List[Any],
        method: str = "custom",
        n_clusters: Optional[int] = None,
        **kwargs
    ) -> List[Theme]:
        # Custom implementation
        ...
```

## Creating a New Workflow

1. **Create the workflow class** in `workflows/`:

```python
# workflows/my_workflow.py
from typing import Dict, Any
from ..analysis_pipeline import AnalysisPipeline

class MyWorkflow:
    """Custom workflow description."""

    def __init__(self, llm_service, db_session=None):
        self.llm_service = llm_service
        self.db_session = db_session

    async def run(self, query: str, **kwargs) -> Dict[str, Any]:
        pipeline = AnalysisPipeline(
            "my_workflow",
            llm_service=self.llm_service,
            db_session=self.db_session
        )

        # Build your pipeline
        pipeline = (
            pipeline
            .expand_query(query, strategy="multi_query")
            .retrieve_segments_by_search(k=100)
            .select_segments(n=20)
            .generate_summaries(template="rag_answer", level="theme")
        )

        # Execute and collect results
        final_context = {}
        async for event in pipeline.execute():
            if event["type"] == "result":
                final_context.update(event.get("data", {}))
            elif event["type"] == "complete":
                break

        return self._format_result(final_context)

    def _format_result(self, context: Dict) -> Dict[str, Any]:
        # Extract and format results
        return {
            "summary": context.get("summaries", {}).get("theme", [None])[0],
            "segment_ids": list(context.get("segment_ids", {}).get("theme", {}).values())[0]
        }
```

2. **Export from `__init__.py`**:

```python
from .my_workflow import MyWorkflow

__all__ = [
    ...,
    'MyWorkflow',
]
```

## API Integration

The workflow system integrates with the `/api/analysis` endpoint via `step_registry.py`:

```python
# POST /api/analysis
{
    "workflow": "simple_rag",
    "query": "climate policy",
    "steps": [
        {"step": "expand_query", "config": {"strategy": "multi_query"}},
        {"step": "retrieve_segments", "config": {"k": 100}},
        {"step": "select_segments", "config": {"n": 20}},
        {"step": "generate_summary", "config": {"template": "rag_answer"}}
    ],
    "filters": {
        "projects": ["Canada"],
        "languages": ["en"]
    }
}
```

## File Structure

```
services/rag/
├── interfaces.py              # Protocol definitions
├── analysis_pipeline.py       # Pipeline framework
├── step_registry.py           # Step metadata and dynamic building
├── segment_retriever.py       # Segment fetching
├── theme_extractor.py         # Theme clustering
├── segment_selector.py        # Segment selection
├── text_generator.py          # LLM text generation
├── quantitative_analyzer.py   # Quantitative metrics
└── workflows/
    ├── __init__.py            # Exports workflows and protocols
    ├── README.md              # This documentation
    ├── simple_rag_workflow.py
    └── hierarchical_summary_workflow.py
```
