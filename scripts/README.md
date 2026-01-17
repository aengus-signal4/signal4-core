# Scripts Directory

Utility shell scripts for deployment and worker management.

## Shell Scripts

| Script | Purpose |
|--------|---------|
| `sync_workers.sh` | Sync code to worker machines |
| `install_uv_workers.sh` | Install uv on worker machines |

## Orchestrator-Driven Tasks

All scheduled processing tasks are now implemented as Python modules in `src/` and invoked via `-m`:

| Module | Purpose | Config Key |
|--------|---------|------------|
| `src.automation.task_creation_manager` | Index sources and create processing tasks | `podcast_index_download`, `youtube_index_download` |
| `src.utils.embedding_hydrator` | Generate embeddings for segments | `embedding_hydration` |
| `src.ingestion.podcast_pipeline` | Monthly podcast chart collection | `podcast_collection` |

See `config/config.yaml` under `scheduled_tasks:` for configuration.

## Running Modules Manually

```bash
# Task creation
python -m src.automation.task_creation_manager --steps index_podcast download_podcast
python -m src.automation.task_creation_manager --project CPRMV --steps download convert

# Embedding hydration
python -m src.utils.embedding_hydrator --batch-size 128

# Podcast pipeline
python -m src.ingestion.podcast_pipeline --phase all
```
