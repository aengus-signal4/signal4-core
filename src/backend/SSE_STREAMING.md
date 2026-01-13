# SSE Stream Event Specification

**Version**: 1.0
**Updated**: 2025-11-14 (Step-based architecture)
**Purpose**: Document the Server-Sent Events (SSE) stream format from the analysis backend

---

## Overview

The backend streams analysis results using Server-Sent Events (SSE). The stream consists of multiple event types that arrive sequentially as the analysis pipeline progresses.

**Architecture**: Each pipeline step emits its own `result` event when complete. The final `complete` event signals pipeline completion (no data).

**Streaming Modes**:
- **Normal mode** (`verbose=false`, default): Emits only `result` and `complete` events
  - **Example**: 5-step pipeline = 6 events (5 result + 1 complete)
- **Verbose mode** (`verbose=true`): Adds `progress` and `partial` events for debugging
  - **Example**: 5-step pipeline = 30+ events (10 progress + ~20 partial + 5 result + 1 complete)

**Recommendation**: Use normal mode for production, verbose mode for debugging.

---

## Enabling Verbose Mode

```json
POST /api/analysis/stream
{
  "query": "What is being said about Quebec?",
  "dashboard_id": "cprmv-practitioner",
  "workflow": "simple_rag",
  "verbose": true  // Set to true for detailed progress updates
}
```

---

## Event Types

### 1. `progress` Events (verbose mode only)

**Purpose**: Indicate step transitions
**When**: At the start/end of each pipeline step

```json
{
  "type": "progress",
  "step": "expand_query",
  "step_index": 0,
  "total_steps": 5,
  "progress": 0.0,
  "message": "Starting expand_query..."
}
```

**Frontend Action**: Update progress bar and step indicator

---

### 2. `partial` Events (verbose mode only)

**Purpose**: Provide incremental progress updates within a step
**When**: During step execution

#### Example: Query Expansion
```json
{
  "type": "partial",
  "step": "expand_query",
  "data": {
    "expanded_query_count": 10,
    "keywords": ["Quebec", "discourse", "media", ...]
  },
  "progress": 0.2,
  "message": "Expanded to 10 query variations"
}
```

#### Example: Segment Retrieval Progress
```json
{
  "type": "partial",
  "step": "retrieve_segments_by_search",
  "data": {
    "query_num": 5,
    "total_queries": 10,
    "segment_count": 459
  },
  "progress": 0.3,
  "message": "Searched query 5/10: 459 segments found"
}
```

#### Example: Baseline Stats
```json
{
  "type": "partial",
  "step": "quantitative_analysis",
  "data": {
    "baseline_segment_count": 4571,
    "analysis_type": "with_baseline"
  },
  "progress": 0.46,
  "message": "Retrieved 4571 baseline segments for comparison"
}
```

**Frontend Action**: Display real-time progress messages (verbose mode only)

---

### 3. `result` Events (STEP OUTPUTS!)

**Purpose**: Contains the output data from each completed pipeline step
**When**: After each step completes successfully

**⚠️ IMPORTANT**: Each step emits its own `result` event with only the data it produced!

#### Step: `expand_query`
```json
{
  "type": "result",
  "step": "expand_query",
  "data": {
    "original_query": "What is being said about Quebec?",
    "expanded_queries": [
      "Instruct: Retrieve relevant passages.\nQuery: Current discussions and media coverage on Quebec's political landscape",
      ...
    ],
    "keywords": ["Quebec", "discourse", "media", ...],
    "expansion_strategy": "multi_query"
  },
  "step_index": 0,
  "total_steps": 5
}
```

#### Step: `retrieve_segments_by_search`
```json
{
  "type": "result",
  "step": "retrieve_segments_by_search",
  "data": {
    "segment_count": 573,
    "segments": [
      {
        "segment_id": 16336453,
        "content_id": 762063,
        "content_id_string": "0BuyAlklqQ4",
        "start_time": 8303.356,
        "end_time": 8394.476,
        "text": "...",
        "channel_name": "La Révolution Culturelle avec Yann Roshdy",
        "title": "Le RETOUR de l'INVASION TIKTOK 10 - 24/10/25",
        "publish_date": "2025-10-24T23:57:48-04:00",
        "similarity": 0.729
      },
      ...
    ]
  },
  "step_index": 1,
  "total_steps": 5
}
```

#### Step: `quantitative_analysis`
```json
{
  "type": "result",
  "step": "quantitative_analysis",
  "data": {
    "quantitative_metrics": {
      "total_segments": 573,
      "unique_videos": 110,
      "unique_channels": 10,
      "channel_distribution": [
        {
          "channel_name": "Ian et Frank",
          "segment_count": 205,
          "percentage": 35.78
        },
        ...
      ],
      "video_distribution": [
        {
          "content_id": "wB38PMlNwd4",
          "title": "Live du mercredi soir !",
          "channel_name": "Ian et Frank",
          "publish_date": "2025-10-29T21:26:12-04:00",
          "segment_count": 26,
          "total_duration_seconds": 1651.81,
          "percentage": 4.54
        },
        ...
      ],
      "temporal_distribution": {
        "granularity": "weekly",
        "earliest_date": "2025-10-16T05:55:09-04:00",
        "latest_date": "2025-11-07T05:00:00-05:00",
        "span_days": 23,
        "time_series": [
          {"period": "2025-W42", "count": 250},
          ...
        ]
      },
      "concentration_metrics": {
        "channel_hhi": 2529.08,
        "channel_interpretation": "concentrated in few channels",
        "video_hhi": 179.06,
        "video_interpretation": "discussed across many episodes",
        "top_3_channels_percentage": 83.42,
        "top_10_videos_percentage": 30.72
      },
      "discourse_centrality": {
        "score": 0.76,
        "interpretation": "Dominant - very widely discussed across the discourse",
        "channel_coverage": 1.111,
        "video_coverage": 0.598,
        "segment_coverage": 0.125,
        "baseline_stats": {
          "total_segments": 4571,
          "unique_videos": 184,
          "unique_channels": 9
        }
      }
    }
  },
  "step_index": 2,
  "total_steps": 5
}
```

#### Step: `select_segments`
```json
{
  "type": "result",
  "step": "select_segments",
  "data": {
    "selected_segments": [
      {
        "segment_id": 16336453,
        "content_id": 762063,
        "content_id_string": "0BuyAlklqQ4",
        "start_time": 8303.356,
        "end_time": 8394.476,
        "text": "...",
        "channel_name": "La Révolution Culturelle avec Yann Roshdy",
        "title": "Le RETOUR de l'INVASION TIKTOK 10 - 24/10/25",
        "publish_date": "2025-10-24T23:57:48-04:00"
      },
      ...
    ],
    "selection_strategy": "diversity"
  },
  "step_index": 3,
  "total_steps": 5
}
```

#### Step: `generate_summaries`
```json
{
  "type": "result",
  "step": "generate_summaries",
  "data": {
    "summaries": {
      "theme": ["Full summary text with {seg_14}, {seg_2}, {seg_9} citations..."]
    },
    "segment_ids": {
      "theme": {
        "simple_rag_summary": [16336453, 16336161, 16336468, ...]
      }
    }
  },
  "step_index": 4,
  "total_steps": 5
}
```

---

### 4. `complete` Event (PIPELINE COMPLETION)

**Purpose**: Signals pipeline completion (no data - just timing/status)
**When**: After all steps complete successfully

```json
{
  "type": "complete",
  "execution_time_ms": 8809,
  "steps_completed": 5,
  "total_steps": 5
}
```

---

## Frontend Implementation Guide

### CORRECT Implementation ✅

```python
# Store results from each step
expanded_queries = []
segments = []
quantitative_metrics = None
selected_segments = []
summaries = {}

# Track progress messages
progress_messages = []

for event in backend.analyze_stream(...):
    event_type = event.get('type')
    step = event.get('step')

    if event_type == 'partial':
        # Use for progress updates ONLY (verbose mode)
        data = event.get('data', {})
        progress_messages.append(event.get('message'))

    elif event_type == 'result':
        # Extract step results
        data = event.get('data', {})

        if step == 'expand_query':
            expanded_queries = data.get('expanded_queries', [])

        elif step == 'retrieve_segments_by_search':
            segments = data.get('segments', [])

        elif step == 'quantitative_analysis':
            quantitative_metrics = data.get('quantitative_metrics', {})

        elif step == 'select_segments':
            selected_segments = data.get('selected_segments', [])

        elif step == 'generate_summaries':
            summaries = data.get('summaries', {})
            segment_ids = data.get('segment_ids', {})

    elif event_type == 'complete':
        # Pipeline complete - all data is already collected
        print(f"Analysis complete in {event['execution_time_ms']}ms")
        break

# Now display results using data from each step's result event
if quantitative_metrics:
    print(f"Total: {quantitative_metrics['total_segments']} segments")
    print(f"Videos: {quantitative_metrics['unique_videos']}")
    print(f"Channels: {quantitative_metrics['unique_channels']}")

    # Display channel distribution
    for channel in quantitative_metrics['channel_distribution']:
        print(f"  {channel['channel_name']}: {channel['segment_count']} ({channel['percentage']}%)")

if summaries:
    summary_text = summaries.get('theme', [''])[0]
    print(f"\nSummary: {summary_text[:200]}...")
```

### INCORRECT Implementation ❌

```python
# DON'T DO THIS!
all_data = {}

for event in backend.analyze_stream(...):
    if event_type == 'complete':
        # Complete event has NO data field anymore!
        all_data = event.get('data', {})  # ← WRONG! Will be empty!

# This will fail because complete event has no data
print(f"Videos: {all_data.get('quantitative_metrics', {}).get('unique_videos', 0)}")  # ← Prints 0!
```

---

## Key Differences: Event Types

**Updated 2025-11-14**: Step-based architecture - each step emits its own `result` event!

| Data | `partial` Event | `result` Event | `complete` Event |
|------|----------------|----------------|------------------|
| **Purpose** | Progress updates | Step output data | Pipeline completion |
| **Frequency** | Multiple per step | One per step | One per pipeline |
| **Data included** | Progress info only | Step-specific output | ❌ **No data** |
| | | | |
| `expanded_queries` | ❌ | ✅ (expand_query step) | ❌ |
| `segments` | ❌ | ✅ (retrieve_segments step) | ❌ |
| `quantitative_metrics` | ✅ (preview) | ✅ (quantitative_analysis step) | ❌ |
| `selected_segments` | ❌ | ✅ (select_segments step) | ❌ |
| `summaries` | ❌ | ✅ (generate_summaries step) | ❌ |
| `execution_time_ms` | ❌ | ❌ | ✅ |

**Key Insight**: Collect data from `result` events (one per step). The `complete` event just signals that the pipeline finished - it has NO data field!

---

## Common Pitfalls

### Pitfall #1: Expecting Data in `complete` Event
**Problem**: Frontend tries to extract all results from the `complete` event's `data` field.

**Result**: `complete` event has NO `data` field - all data comes from individual step `result` events.

**Solution**: Collect data from `result` events as they arrive. Use `complete` event only to detect pipeline completion.

### Pitfall #2: Ignoring `result` Events
**Problem**: Code only processes `partial` and `complete` events, ignoring `result` events.

**Result**: No data is collected - all step outputs are lost.

**Solution**: Always handle `result` events to extract step outputs.

### Pitfall #3: Not Using `step` Field
**Problem**: Treating all `result` events the same without checking the `step` field.

**Result**: Data from different steps gets mixed up or overwritten.

**Solution**: Check `event.get('step')` to determine which step completed and where to store the data:
```python
if event_type == 'result':
    step = event.get('step')
    data = event.get('data', {})

    if step == 'quantitative_analysis':
        quantitative_metrics = data.get('quantitative_metrics', {})
    elif step == 'generate_summaries':
        summaries = data.get('summaries', {})
```

---

## Summary

**Normal Mode (default, `verbose=false`):**
- 5× `result` events (one per step) - **Extract all data from these!**
- 1× `complete` event (pipeline done)
- **Total: 6 events**

**Verbose Mode (`verbose=true`):**
- 10× `progress` events (step start/end for 5 steps)
- ~20× `partial` events (real-time progress within steps)
- 5× `result` events (one per step) - **Extract all data from these!**
- 1× `complete` event (pipeline done)
- **Total: 30+ events**

**Golden Rule**: Collect data from `result` events (one per step). The `complete` event has NO data field - it just signals the pipeline finished.

---

## Related Documentation

- **Backend README**: `/src/backend/README.md` - API endpoints and configuration
- **RAG Infrastructure**: `/src/backend/app/services/rag/README.md` - Pipeline architecture and available steps
- **Step Registry**: `/src/backend/app/services/rag/step_registry.py` - All available steps with parameters
- **Workflows**: `/src/backend/app/config/workflows.py` - Predefined workflow templates
- **Analysis Pipeline**: `/src/backend/app/services/rag/analysis_pipeline.py` - Pipeline implementation
- **Frontend Spec**: `/signal4.ca/SSE_EVENT_SPEC.md` - Frontend-focused version of this document
