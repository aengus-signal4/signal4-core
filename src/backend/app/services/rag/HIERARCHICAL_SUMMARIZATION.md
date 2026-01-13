# Hierarchical Summarization System

A flexible, project-agnostic system for generating hierarchical summaries with citation tracking across user-defined groups and themes.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Citation System](#citation-system)

---

## Overview

The Hierarchical Summarization System enables researchers to generate structured, citation-tracked summaries of discourse patterns across large corpora of audio-visual content. It supports:

- **Flexible Grouping**: User-defined groups based on channel metadata, keywords, language, projects
- **Theme Discovery**: Automated clustering (UMAP+HDBSCAN) or predefined themes
- **Two-Pass Summarization**: Theme summaries → Meta-synthesis
- **Citation Tracking**: Every factual claim linked to source segments
- **Project-Agnostic**: Works with any content collection

### Key Features

✅ **No Hardcoded Groups**: All groupings defined via API parameters
✅ **Metadata-Driven**: Leverages JSONB metadata for flexible filtering
✅ **Citation Integrity**: Validates all citations against source segments
✅ **Scalable**: Handles millions of segments via FAISS indexes
✅ **Reproducible**: Config hashes ensure identical inputs → identical outputs

---

## Architecture

### Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Group Definition (DynamicGrouper)                      │
│  ─────────────────────────────────────────────────────────────  │
│  • User provides filters: channel_urls, keywords, language      │
│  • System queries database and builds segment groups            │
│  • Result: GroupResult with segment_ids, metadata               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Theme Discovery (TopicDiscovery)                       │
│  ─────────────────────────────────────────────────────────────  │
│  • Per-group clustering: UMAP → HDBSCAN                         │
│  • OR: Semantic search for predefined themes                    │
│  • Result: List of ThemeCluster with representative segments    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Segment Sampling (SegmentSampler)                      │
│  ─────────────────────────────────────────────────────────────  │
│  • Sample ~20 representative segments per theme                 │
│  • Strategies: balanced, top_similarity, diverse_channels       │
│  • Result: List of SampledSegment with metadata                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Theme Summarization (ThemeSummarizer)                  │
│  ─────────────────────────────────────────────────────────────  │
│  • Generate citation IDs: [G{group}-T{theme}-S{segment}]        │
│  • Build LLM prompt with numbered segments + citations          │
│  • Generate 7-8 paragraph summary with embedded citations       │
│  • Validate citations against segment IDs                       │
│  • Result: ThemeSummary with summary_text + citations map       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Meta-Summarization (MetaSummarizer)                    │
│  ─────────────────────────────────────────────────────────────  │
│  • Aggregate all theme summaries                                │
│  • Build meta-prompt with ALL theme summaries + citations       │
│  • Generate 10-paragraph cross-theme synthesis                  │
│  • REUSE existing citations only (no new citations created)     │
│  • Validate citation integrity                                  │
│  • Result: MetaSummary with synthesis_text + all citations      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. CitationManager

**Purpose**: Manages citation ID generation, parsing, and validation.

**Key Methods**:
```python
citation_manager = CitationManager(citation_format="[G{group_id}-T{theme_id}-S{segment_id}]")

# Generate citation
citation = citation_manager.generate(
    group_id="religious_fr",
    theme_id="T5",
    segment_id=12847
)
# Returns: "[G_religious_fr-T5-S12847]"

# Parse citation
parsed = citation_manager.parse("[G_religious_fr-T5-S12847]")
# Returns: Citation(group_id="religious_fr", theme_id="T5", segment_id=12847)

# Extract all citations from text
citations = citation_manager.extract_all(summary_text)

# Validate citations
valid = citation_manager.validate(citation, valid_segment_ids)
```

**Citation Format**: Fully customizable via format string with `{group_id}`, `{theme_id}`, `{segment_id}` placeholders.

---

### 2. DynamicGrouper

**Purpose**: Dynamically filter and group segments based on user criteria.

**Filter Options**:
```python
filter_config = GroupFilter(
    channel_urls=["https://youtube.com/@channel1", "https://youtube.com/@channel2"],
    keywords=["religieux", "anti-trans"],  # Searches meta_data JSONB
    language="fr",
    projects=["CPRMV"],
    meta_data_query={"explicit": True}  # Custom JSONB queries
)

grouper = DynamicGrouper(session)
group_result = grouper.build_group(
    group_id="religious_fr",
    group_name="Religious French Content",
    filter_config=filter_config,
    time_window_days=30
)
```

**GroupResult**:
```python
{
    'group_id': 'religious_fr',
    'group_name': 'Religious French Content',
    'segment_ids': [12847, 12848, ...],
    'channel_urls': ['https://youtube.com/@channel1', ...],
    'segment_count': 1543,
    'date_range': ('2024-01-01', '2024-01-31'),
    'metadata': {
        'unique_content_items': 87,
        'unique_channels': 5,
        'avg_segment_length': 45.2,
        'total_duration': 69743.5
    }
}
```

---

### 3. TopicDiscovery (Extended)

**Purpose**: Discover themes via clustering, scoped to groups.

**New Method**: `discover_topics_for_group()`

```python
topic_discovery = TopicDiscovery(
    search_service=search_service,
    time_window_days=30,
    min_cluster_size=15,
    umap_n_components=50
)

# Discover themes for a specific group
themes = topic_discovery.discover_topics_for_group(
    group_segment_ids=group_result.segment_ids,
    group_id="religious_fr"
)
```

**Returns**: List of `ClusterMetrics` with:
- `cluster_id`: Theme identifier
- `segment_ids`: All segments in this theme
- `representative_segment_ids`: Top 5 most representative
- `breadth`, `intensity`, `coherence`, `recency`: Scoring components
- `score`: Combined relevance score

---

### 4. SegmentSampler

**Purpose**: Smart sampling of representative segments for summarization.

**Sampling Strategies**:

```python
sampler = SegmentSampler(max_samples=20, strategy="balanced")

# Balanced: 50% top similarity, 25% channel diversity, 25% temporal spread
samples = sampler.sample(segments, similarity_scores)

# Top similarity only
sampler = SegmentSampler(max_samples=20, strategy="top_similarity")

# Maximize channel diversity
sampler = SegmentSampler(max_samples=20, strategy="diverse_channels")

# Maximize temporal spread
sampler = SegmentSampler(max_samples=20, strategy="temporal_spread")
```

**SampledSegment** includes:
- `segment_id`, `text`, `channel_url`, `publish_date`
- `similarity_score`: Similarity to theme centroid
- `sampling_reason`: Why this segment was selected

---

### 5. ThemeSummarizer

**Purpose**: Generate first-pass theme summaries with embedded citations.

**Usage**:
```python
summarizer = ThemeSummarizer(
    llm_service=llm_service,
    citation_manager=citation_manager,
    max_context_segments=20
)

theme_summary = summarizer.generate_summary(
    group_id="religious_fr",
    theme_id="T5",
    theme_name="Gender Identity in Schools",
    sampled_segments=sampled_segments
)
```

**ThemeSummary Output**:
```python
{
    'group_id': 'religious_fr',
    'theme_id': 'T5',
    'theme_name': 'Gender Identity in Schools',
    'summary_text': '...many podcasters argue... [G_religious_fr-T5-S12847]...',
    'citations': {
        '[G_religious_fr-T5-S12847]': {
            'segment_id': 12847,
            'text': '...',
            'channel_name': '...',
            'publish_date': '2024-01-15'
        },
        ...
    },
    'segment_count': 20,
    'invalid_citations': [],
    'generation_time_ms': 3542
}
```

---

### 6. MetaSummarizer

**Purpose**: Generate second-pass meta-synthesis across themes.

**Synthesis Types**:
- `cross_theme`: Identify patterns across themes (default)
- `cross_group`: Compare different groups
- `temporal`: Track evolution over time

**Usage**:
```python
meta_summarizer = MetaSummarizer(
    llm_service=llm_service,
    citation_manager=citation_manager
)

meta_summary = meta_summarizer.generate_synthesis(
    theme_summaries=[theme1, theme2, theme3, ...],
    synthesis_type="cross_theme"
)
```

**Key Constraint**: Meta-summarizer can ONLY reuse citations from theme summaries—it cannot create new citation IDs.

---

## Usage Guide

### Example 1: Single Group, Clustering-Based Themes

```python
# POST /api/summary/hierarchical/generate

{
  "time_window_days": 30,
  "groupings": [
    {
      "group_id": "religious_fr",
      "group_name": "Religious French Content",
      "filter": {
        "keywords": ["religieux", "chrétien"],
        "language": "fr"
      }
    }
  ],
  "theme_discovery_method": "clustering",
  "clustering_params": {
    "min_cluster_size": 15,
    "umap_n_components": 50
  },
  "samples_per_theme": 20,
  "generate_meta_summary": true,
  "synthesis_type": "cross_theme"
}
```

**Response**:
```json
{
  "success": true,
  "summary_id": "hs_a1b2c3d4e5f6",
  "theme_summaries": [
    {
      "theme_id": "T0",
      "theme_name": "Theme 0",
      "summary_text": "...[G_religious_fr-T0-S12847]...",
      "citations": {...},
      "segment_count": 20
    },
    ...
  ],
  "meta_summary": {
    "synthesis_text": "...synthesis across all themes...",
    "theme_count": 5
  },
  "total_themes": 5,
  "total_groups": 1,
  "total_citations": 87,
  "processing_time_ms": 45230
}
```

---

### Example 2: Multiple Groups, Cross-Group Comparison

```python
{
  "time_window_days": 30,
  "groupings": [
    {
      "group_id": "religious_en",
      "group_name": "Religious English Content",
      "filter": {
        "keywords": ["religious", "christian"],
        "language": "en"
      }
    },
    {
      "group_id": "religious_fr",
      "group_name": "Religious French Content",
      "filter": {
        "keywords": ["religieux", "chrétien"],
        "language": "fr"
      }
    }
  ],
  "theme_discovery_method": "clustering",
  "samples_per_theme": 20,
  "generate_meta_summary": true,
  "synthesis_type": "cross_group"  // Compare EN vs FR
}
```

**Result**: Meta-summary comparing how English vs. French religious content frames similar themes differently.

---

### Example 3: Predefined Themes (CPRMV Use Case)

```python
{
  "time_window_days": 30,
  "groupings": [
    {
      "group_id": "masculinist",
      "group_name": "Masculinist Content",
      "filter": {
        "keywords": ["masculinist", "men's rights"]
      }
    }
  ],
  "theme_discovery_method": "predefined",
  "predefined_themes": [
    {
      "theme_id": "A1",
      "theme_name": "Biological Essentialism",
      "query_variations": [
        "Men and women are biologically different",
        "Gender is determined by biology",
        ...
      ]
    },
    {
      "theme_id": "A2",
      "theme_name": "Anti-Feminism",
      "query_variations": [
        "Feminism has gone too far",
        "Men are oppressed by feminism",
        ...
      ]
    }
  ],
  "samples_per_theme": 20
}
```

---

## API Reference

### POST `/api/summary/hierarchical/generate`

**Request Body**:
```typescript
{
  time_window_days: number;  // 1-365
  groupings: Array<{
    group_id: string;
    group_name: string;
    filter: {
      channel_urls?: string[];
      keywords?: string[];
      language?: string;
      projects?: string[];
      meta_data_query?: object;
    };
  }>;
  theme_discovery_method: "clustering" | "predefined";
  predefined_themes?: Array<{theme_id, theme_name, query_variations}>;
  clustering_params?: {
    min_cluster_size?: number;
    umap_n_components?: number;
  };
  samples_per_theme: number;  // 5-50
  citation_format?: string;  // Default: "[G{group_id}-T{theme_id}-S{segment_id}]"
  generate_meta_summary: boolean;
  synthesis_type: "cross_theme" | "cross_group" | "temporal";
}
```

**Response**:
```typescript
{
  success: boolean;
  summary_id: string;
  theme_summaries: Array<ThemeSummary>;
  meta_summary: MetaSummary | null;
  total_themes: number;
  total_groups: number;
  total_citations: number;
  processing_time_ms: number;
  config_hash: string;
}
```

---

### GET `/api/summary/hierarchical/citation/{citation_id}`

Retrieve full segment metadata for a citation.

**Example**: `GET /api/summary/hierarchical/citation/[G_religious_fr-T5-S12847]`

**Response**:
```json
{
  "success": true,
  "citation_id": "[G_religious_fr-T5-S12847]",
  "segment_id": 12847,
  "text": "Full segment text here...",
  "start_time": 123.5,
  "end_time": 145.2,
  "channel_name": "Le Lucide Podcast",
  "title": "Episode 42: Gender Ideology in Schools",
  "publish_date": "2024-01-15T10:30:00"
}
```

---

## Citation System

### Citation Format

Default: `[G{group_id}-T{theme_id}-S{segment_id}]`

**Examples**:
- `[G_religious_fr-T5-S12847]`
- `[G_masculinist_en-T12-S98234]`

### Custom Formats

Fully customizable via `citation_format` parameter:

```python
# Alternative format: (Group:Theme:Segment)
citation_format = "(G{group_id}:T{theme_id}:S{segment_id})"
# Result: "(G_religious_fr:T5:S12847)"

# Compact format: G-T-S
citation_format = "{group_id}-{theme_id}-{segment_id}"
# Result: "religious_fr-T5-12847"
```

### Citation Validation

**Automatic validation**:
- Theme summaries: Validates all citations against sampled segments
- Meta-summaries: Validates against all theme citations
- Invalid citations logged in `invalid_citations` array

**Manual validation**:
```python
# Theme summary validation
report = summarizer.validate_summary_citations(theme_summary)
# Returns: {total_citations, valid_citations, invalid_citations, segment_coverage_rate, ...}

# Meta-summary validation
report = meta_summarizer.validate_synthesis_citations(meta_summary)
# Returns: {total_citations, themes_cited, groups_cited, citations_per_theme, ...}
```

---

## Database Schema

### HierarchicalSummary Table

```sql
CREATE TABLE hierarchical_summaries (
    id SERIAL PRIMARY KEY,
    summary_type VARCHAR(50) NOT NULL,  -- 'theme' or 'meta'
    summary_id VARCHAR(255) UNIQUE NOT NULL,
    config_hash VARCHAR(64) NOT NULL,
    summary_data JSONB NOT NULL,  -- Full summary + citations
    time_window_days INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    INDEX idx_hierarchical_summary_config (config_hash, summary_type),
    INDEX idx_hierarchical_summary_id (summary_id)
);
```

**Storage**: Summaries stored as JSONB for flexibility. No separate Group/Theme tables—fully dynamic.

---

## Configuration & Caching

### Config Hash

Every request generates a unique config hash:
```python
config_hash = md5(request.model_dump_json()).hexdigest()
```

**Purpose**:
- Enables result caching
- Ensures reproducibility
- Identifies identical requests

### LLM Caching

Both ThemeSummarizer and MetaSummarizer use `LLMCache` for response caching:
- Theme summaries cached by theme_id + group_id + segments hash
- Meta-summaries cached by theme_summaries hash
- TTL: Permanent (no expiration by default)

---

## Performance Considerations

### Scalability

- **FAISS Indexes**: Handle millions of vectors efficiently
- **Group-Scoped Clustering**: Reduces dimensionality of UMAP/HDBSCAN
- **Batch Processing**: Reuse search indexes across themes
- **Caching**: Avoids redundant LLM calls

### Optimization Tips

1. **Limit time windows**: Smaller windows = faster clustering
2. **Adjust min_cluster_size**: Larger = fewer, coarser themes
3. **Reduce samples_per_theme**: Fewer segments = faster LLM generation
4. **Use predefined themes**: Skip clustering entirely for known themes

### Estimated Processing Times

| Scale | Segments | Groups | Themes | Time |
|-------|----------|--------|--------|------|
| Small | 10K | 1 | 5 | 2-5 min |
| Medium | 100K | 2 | 10 | 5-15 min |
| Large | 1M | 5 | 25 | 15-45 min |

---

## Troubleshooting

### No themes discovered

**Cause**: min_cluster_size too large or segments too sparse

**Solution**: Reduce `min_cluster_size` or increase `time_window_days`

### Invalid citations in summary

**Cause**: LLM generated non-existent citation IDs

**Solution**:
- Check `invalid_citations` array in response
- Refine LLM prompt (adjust temperature)
- Validate and filter before storage

### Empty groups

**Cause**: Filters too restrictive or no matching content

**Solution**:
- Validate filters with `validate_group_filters()`
- Check channel_urls exist in database
- Broaden keyword filters

---

## Future Enhancements

- [ ] **Export Formats**: Markdown, HTML, PDF generation
- [ ] **Async Job System**: Background processing for large requests
- [ ] **Citation Enrichment**: Add audio timestamps, speaker attribution
- [ ] **Interactive Visualization**: Web UI for exploring summaries
- [ ] **Multi-Language Support**: Cross-language theme discovery
- [ ] **Incremental Updates**: Update summaries without full regeneration

---

## Contributing

See `CONTRIBUTING.md` for development guidelines.

## License

See `LICENSE.md` for license information.
