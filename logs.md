# Core Logs

## Base Path

`/Users/signal4/logs/content_processing/`

## Orchestrator Logs (Top-level)

| Log File | Purpose |
|----------|---------|
| `orchestrator_debug.log` | Main orchestrator debug log |
| `orchestrator_errors.log` | Orchestrator errors |
| `task_completions.log` | All task completions across workers |

```bash
# Tail orchestrator errors
tail -f /Users/signal4/logs/content_processing/orchestrator_errors.log

# Search task completions
grep "transcribe" /Users/signal4/logs/content_processing/task_completions.log | tail -20
```

## Per-Worker Logs

Logs are organized by worker name (e.g., `worker6/`):

### Backend Logs

| Log File | Purpose |
|----------|---------|
| `worker6_backend.log` | General backend operations |
| `worker6_backend.cached_workflow_executor.log` | RAG workflow execution |
| `worker6_backend.llm.log` | LLM calls and responses |
| `worker6_backend.pgvector_search.log` | Vector similarity search |
| `worker6_backend.query_service.log` | Query endpoint processing |
| `worker6_backend.embedding.log` | Embedding generation |

### Processing Logs

| Log File | Purpose |
|----------|---------|
| `worker6_transcribe.log` | Transcription tasks |
| `worker6_diarize.log` | Diarization tasks |
| `worker6_stitch.log` | Stitch tasks |
| `worker6_segment_embed.log` | Segment embedding tasks |
| `worker6_cleanup.log` | Cleanup tasks |

## Quick Commands

```bash
# Tail RAG workflow log
tail -f /Users/signal4/logs/content_processing/worker6/worker6_backend.cached_workflow_executor.log

# Tail multiple backend logs
tail -f /Users/signal4/logs/content_processing/worker6/worker6_backend.{cached_workflow_executor,pgvector_search,llm}.log

# Search for errors across backend logs
grep -i "error\|exception" /Users/signal4/logs/content_processing/worker6/worker6_backend.*.log | tail -50

# View last 100 lines of workflow executor
tail -100 /Users/signal4/logs/content_processing/worker6/worker6_backend.cached_workflow_executor.log
```

## Remote Worker Logs

Workers store logs at the same path on each machine. SSH to access:

| Worker | IP | SSH Command |
|--------|-----|-------------|
| worker0 | 10.0.0.34 | `ssh signal4@10.0.0.34` |
| worker3 | 10.0.0.203 | `ssh signal4@10.0.0.203` |
| worker4 | 10.0.0.51 | `ssh signal4@10.0.0.51` |
| worker5 | 10.0.0.209 | `ssh signal4@10.0.0.209` |
| worker6 | 10.0.0.4 | Local (head node) |

```bash
# Tail remote worker logs
ssh signal4@10.0.0.34 'tail -f /Users/signal4/logs/content_processing/worker0/worker0_transcribe.log'

# Copy logs locally
scp signal4@10.0.0.34:/Users/signal4/logs/content_processing/worker0/*.log ./
```

## Log Configuration

Logging is configured in `core/config/config.yaml` under the `logging` section:

```yaml
logging:
  level: INFO
  base_path: "/Users/signal4/logs/content_processing"
```

Logger implementation: `core/src/utils/logger.py`
