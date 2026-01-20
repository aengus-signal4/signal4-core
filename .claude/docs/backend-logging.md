# Backend Logging System

The backend uses a hybrid logging system (`src/backend/app/utils/backend_logger.py`).

## Two Tiers

1. **Workflow log**: Key milestones → console + `logs/backend/workflow.log`
2. **Component logs**: Debug detail → per-component files (no console noise)

## Console Output Format

Clean, scannable workflow trace:
```
2024-01-20 14:30:15 [health] [request] Starting discourse_summary (no query)
2024-01-20 14:30:15 [health] [cache] Cache MISS - running full pipeline
2024-01-20 14:30:16 [health] [1/5] expand_query: 10 variations (450ms)
2024-01-20 14:30:18 [health] [2/5] retrieve_segments: 342 segments (1892ms)
2024-01-20 14:30:31 [health] [complete] Analysis complete (15892ms)
```

## Log File Structure

```
logs/backend/
  workflow.log              # Milestones (console mirrors this)
  analysis_router.log       # Component debug (file only)
  cached_workflow_executor.log
  embedding_service.log
  pgvector_search.log
  llm_service.log
  ... (one per component)
```

## Usage

```python
# Component logger (file only, DEBUG+)
from ..utils.backend_logger import get_logger
logger = get_logger("my_component")
logger.debug("Internal detail...")  # File only
logger.info("Processing...")        # File only

# Workflow events (console + file)
from ..utils.backend_logger import log_request_start, log_step_complete
log_request_start(dashboard_id, workflow, query)
log_step_complete(dashboard_id, step_num, total_steps, step_name, duration_ms, details)
```

## Workflow Functions

| Function | Purpose |
|----------|---------|
| `log_request_start(dashboard_id, workflow, query)` | Request entry |
| `log_request_complete(dashboard_id, cached)` | Request completion with timing |
| `log_cache_hit(dashboard_id, level, age_info)` | Cache hit events |
| `log_cache_miss(dashboard_id, reason)` | Cache miss events |
| `log_step_complete(dashboard_id, step_num, total, name, ms, details)` | Pipeline steps |
| `log_error(dashboard_id, error, step)` | Error events |
