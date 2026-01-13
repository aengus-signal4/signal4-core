# RAG Analysis Infrastructure

**Step-based pipeline system for semantic search, theme extraction, and LLM-powered summarization**

## Overview

This infrastructure provides a **composable, declarative pipeline framework** for building RAG (Retrieval-Augmented Generation) workflows. Pipelines are defined as sequences of steps that can be configured via JSON and executed with SSE streaming for progressive results.

**Key Features:**
- **Step-based architecture**: Each step emits its own `result` event when complete
- **Declarative workflows**: Define pipelines via JSON configuration (no code changes)
- **SSE streaming**: Two modes (normal: result events only, verbose: all events)
- **Flexible composition**: Add custom steps, combine existing steps, create new workflows
- **Layer-based design**: Data retrieval → Analysis primitives → Workflow orchestration

## Architecture

### 3-Layer Modular Design

**Layer 1: Data Retrieval** (Core data access and indexing)
- `SegmentRetriever` - Unified segment fetching with flexible filters (database-first)
- `QueryParser` - LLM-based query expansion (multi_query, query2doc, theme_queries)
- `SmartRetriever` - High-level retrieval combining query parsing + semantic search

**Layer 2: Analysis Primitives** (Reusable analysis components)
- `ThemeExtractor` - Theme discovery via clustering (UMAP + HDBSCAN) with adaptive validation
- `SegmentSelector` - Weighted segment sampling (diversity, centrality, recency, quality)
- `TextGenerator` - LLM text generation with prompt templates and batch processing
- `QuantitativeAnalyzer` - Volume metrics, channel distribution, discourse centrality

**Layer 3: Workflows** (Composable analysis pipelines)
- `AnalysisPipeline` - Declarative workflow orchestration with step registry
- Streaming support via Server-Sent Events (SSE) for progressive results
- Multiple predefined workflows (see [workflows.py](../../config/workflows.py))

### Pipeline Execution Flow

```
User Request (JSON)
    ↓
AnalysisPipeline + Step Registry
    ↓
Step 1: expand_query → emit result event
    ↓
Step 2: retrieve_segments → emit result event
    ↓
Step 3: quantitative_analysis → emit result event
    ↓
Step 4: select_segments → emit result event
    ↓
Step 5: generate_summary → emit result event
    ↓
Pipeline Complete → emit complete event (no data)
```

### SSE Event Architecture

**Normal Mode** (`verbose=false`, default):
- Each step emits one `result` event with its output data
- Final `complete` event signals pipeline completion (no data, just timing)
- **Example**: 5-step pipeline = 6 events total (5 result + 1 complete)

**Verbose Mode** (`verbose=true`, debugging):
- Adds `progress` events (step start/end)
- Adds `partial` events (real-time progress within steps)
- **Example**: 5-step pipeline = 30+ events total

**Key Insight**: Frontend collects data from `result` events as they arrive. The `complete` event just signals "pipeline done" with no data field.

## Available Pipeline Steps

All steps are registered in [step_registry.py](step_registry.py) and can be composed into custom workflows.

### Query Expansion
**`expand_query`** - Expand query into multiple variations for improved recall
- **Strategies**: `multi_query`, `query2doc`, `theme_queries`, `stance_variation`
- **Output**: `original_query`, `expanded_queries`, `keywords`, `expansion_strategy`

### Segment Retrieval
**`retrieve_segments`** - Retrieve segments using pgvector semantic search
- **Parameters**: `k` (results per query), `time_window_days`, `projects`, `languages`, `channels`
- **Output**: `segment_count`, `segments` (with metadata: text, channel, video, timestamps)

### Theme Extraction
**`extract_themes`** - Discover latent themes via UMAP + HDBSCAN clustering
- **Parameters**: `method` (hdbscan/kmeans), `n_clusters`, `min_cluster_size`, `min_samples`
- **Output**: `themes` (clusters with size, coherence, segments)

**`extract_subthemes`** - Extract sub-themes within each theme
- **Parameters**: `n_subthemes`, `min_cluster_size`, `require_valid_clusters`, `min_silhouette_score`
- **Output**: `subthemes` (nested clusters per theme)

### Quantitative Analysis
**`quantitative_analysis`** - Generate volume, distribution, and centrality metrics
- **Parameters**: `include_baseline` (for discourse centrality), `time_window_days`
- **Output**: `quantitative_metrics`:
  - `total_segments`, `unique_videos`, `unique_channels`
  - `channel_distribution`, `video_distribution`, `temporal_distribution`
  - `concentration_metrics` (HHI, top-N percentages)
  - `discourse_centrality` (score, interpretation, coverage, baseline stats)

### Segment Selection
**`select_segments`** - Select diverse subset of segments
- **Strategies**: `diversity`, `balanced`, `recency`
- **Parameters**: `n` (number to select per theme)
- **Output**: `selected_segments` (with full metadata)

### Summary Generation
**`generate_summary`** - Generate LLM summaries with citations
- **Parameters**: `template` (rag_answer, theme_summary, meta_summary), `level` (theme/subtheme/meta), `model`, `temperature`, `max_tokens`, `max_concurrent`
- **Output**: `summaries` (text with citations), `segment_ids` (citation mapping)

### Utility Steps
**`group_by`** - Group segments by field (language, channel, etc.)
- **Parameters**: `field` (field name to group by)
- **Output**: Grouped segment collections

**`custom_step`** - Add custom processing step
- **Parameters**: `name`, `func` (async callable), `**params`
- **Output**: Custom (depends on function)

## Predefined Workflows

Defined in [workflows.py](../../config/workflows.py) - reference by name in API requests.

### `simple_rag`
**Purpose**: Simple RAG workflow for quick query answering
**Steps**: expand_query → retrieve_segments → quantitative_analysis → select_segments → generate_summary
**Use case**: Dashboard search, quick discourse analysis

### `search_only`
**Purpose**: Retrieve segments without summarization
**Steps**: expand_query → retrieve_segments
**Use case**: Raw search results, data export

### `hierarchical_summary`
**Purpose**: Theme-based analysis with multi-level summaries
**Steps**: retrieve_segments → extract_themes → quantitative_analysis → select_segments → generate_summary (theme) → generate_summary (meta)
**Use case**: Comprehensive discourse analysis, trend reports

### `hierarchical_with_subthemes`
**Purpose**: Deep analysis with themes, sub-themes, and multi-level summaries
**Steps**: retrieve_segments → extract_themes → extract_subthemes → quantitative_analysis → select_segments → generate_summary (theme + subtheme + meta)
**Use case**: In-depth research, complex discourse mapping

### `deep_analysis`
**Purpose**: Multi-stance analysis with baseline comparison
**Steps**: expand_query (stance_variation) → retrieve_segments → extract_themes → quantitative_analysis (with baseline) → select_segments → generate_summary (theme + meta)
**Use case**: Controversial topics, stance diversity analysis

## Adding Custom Steps

**1. Implement method on `AnalysisPipeline`** (analysis_pipeline.py):
```python
def sentiment_analysis(self, model="vader") -> "AnalysisPipeline":
    self.steps.append(("sentiment_analysis", {"model": model}))
    return self
```

**2. Add execution logic in `_execute_step_stream()`**:
```python
elif step_type == "sentiment_analysis":
    segments = context.get("segments", [])
    # Process sentiment...
    context["sentiment_scores"] = scores
    yield {"type": "result", "data": context}
```

**3. Define result extraction in `_extract_step_results()`**:
```python
elif step_type == "sentiment_analysis":
    return {
        "sentiment_scores": context.get("sentiment_scores", {}),
        "average_sentiment": context.get("average_sentiment", 0.0)
    }
```

**4. Register in `step_registry.py`**:
```python
"sentiment_analysis": StepMetadata(
    name="sentiment_analysis",
    description="Analyze sentiment of segments",
    parameters={"model": {"type": "string", "default": "vader"}},
    method_name="sentiment_analysis"
)
```

**5. Use in workflows** (workflows.py):
```python
WORKFLOWS["rag_with_sentiment"] = [
    {"step": "retrieve_segments", "config": {}},
    {"step": "sentiment_analysis", "config": {"model": "vader"}},
    {"step": "generate_summary", "config": {}}
]
```

## Topic Scoring

Topics are ranked by a weighted combination of four metrics:

```python
score = α * breadth       # Unique channels discussing topic
      + β * intensity     # Segments per day
      + γ * coherence     # Cluster tightness (HDBSCAN score)
      + δ * recency       # Time-weighted (recent = higher)
```

**Default weights:**
- Breadth: 0.35 (channel diversity is important)
- Intensity: 0.25 (volume of discussion)
- Coherence: 0.25 (cluster quality)
- Recency: 0.15 (time relevance)

### Metric Details

**Breadth** - How many unique channels discuss this topic
- Normalized by expected max (~20 channels for 7d window)
- Higher = more significant/widespread topic

**Intensity** - Concentration of discussion (segments/day)
- Normalized by expected max (~50 segments/day for hot topics)
- Higher = "hotter" topic, actively being discussed

**Coherence** - Cluster tightness via HDBSCAN membership probabilities
- Average probability across cluster members (0-1)
- Higher = well-defined topic vs noise

**Recency** - Time-weighted score with exponential decay
- Half-life = window_days / 2
- Recent content weighted higher than old content

## Usage

### Quick Test

```bash
# Discover topics from past 7 days
python scripts/test_topic_discovery.py --days 7 --top-n 10

# Save results to JSON
python scripts/test_topic_discovery.py --days 7 --output results.json

# Fast mode (skip LLM labeling)
python scripts/test_topic_discovery.py --days 7 --skip-labels
```

### Python API

```python
from app.config.dashboard_config import load_dashboard_config
from app.services.search_service import SearchService
from app.services.llm_service import LLMService
from app.services.rag.topic_discovery import TopicDiscovery
from app.services.rag.topic_labeler import TopicLabeler

# Initialize services
config = load_dashboard_config('cprmv-practitioner')
search_service = SearchService('cprmv-practitioner', config)
llm_service = LLMService(config, 'cprmv-practitioner')

# Discover topics
topic_discovery = TopicDiscovery(
    search_service=search_service,
    time_window_days=7,
    min_cluster_size=15
)
topics = topic_discovery.discover_topics()

# Top topic by score
top_topic = topics[0]
print(f"Score: {top_topic.score:.3f}")
print(f"Size: {top_topic.size} segments")
print(f"Channels: {len(top_topic.channels)}")

# Get sample texts for labeling
samples = topic_discovery.get_cluster_texts(top_topic.cluster_id)

# Generate label
labeler = TopicLabeler(llm_service)
label = labeler.label_topic(top_topic.cluster_id, samples)
print(f"Topic: {label.topic_name}")
print(f"Description: {label.topic_description}")
```

### Configuration Options

**TopicDiscovery parameters:**
- `time_window_days` (7) - Time window for analysis
- `min_cluster_size` (15) - Minimum HDBSCAN cluster size
- `min_samples` (5) - HDBSCAN min samples for core points
- `umap_n_components` (50) - UMAP target dimensionality
- `scoring_weights` - Dict with {breadth, intensity, coherence, recency}

**TopicLabeler parameters:**
- Uses existing LLMService for caching and API calls
- Temperature: 0.3 (consistent labeling)
- Max tokens: 1000

## Dependencies

**Clustering:**
- `umap-learn` - Dimensionality reduction
- `hdbscan` - Density-based clustering
- `faiss-cpu` - Vector search (already installed)

**LLM:**
- `requests` - xAI Grok API calls (already installed)
- Uses existing backend LLMService infrastructure

Install clustering dependencies:
```bash
pip install umap-learn hdbscan
```

## Performance

### Caching Strategy

**SearchService:**
- Memory + disk caching of FAISS indexes
- 7-day windows: 1hr expiry
- 30-day windows: 24hr expiry
- Stale cache returns immediately + triggers async rebuild

**TopicLabeler:**
- LLM responses cached permanently via LLMService cache
- Cache key based on sample texts (deterministic)

### Timing Estimates (7-day window, ~50k segments)

- FAISS index build: ~3-5s (first time)
- FAISS index load: ~100ms (cached)
- UMAP reduction: ~10-15s
- HDBSCAN clustering: ~5-10s
- LLM labeling: ~2-3s per topic (cached: <10ms)

**Total:** ~20-30s for first run, ~15s for cached index

## Next Steps

To integrate into backend API:

1. Create FastAPI router at `app/routers/topics.py`
2. Add endpoint: `POST /api/topics/discover`
3. Add endpoint: `GET /api/topics/{topic_id}/summary`
4. Consider background task for periodic topic updates
5. Add to frontend dashboard

Example endpoint:
```python
@router.post("/discover")
async def discover_topics(
    time_window_days: int = 7,
    top_n: int = 10
):
    # Run discovery pipeline
    # Return labeled topics with scores
    pass
```

## Notes

- Uses **1024-dim embeddings** (100% coverage) not 2000-dim (11.4% coverage)
- Designed for **short-term analysis** (7-30 days), scoring reflects this
- HDBSCAN may label some segments as noise (-1 cluster) - this is expected
- Topic quality improves with larger time windows and lower min_cluster_size
