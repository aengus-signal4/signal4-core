# Automation - Scheduled Task System

Config-driven scheduled task execution system.

## Overview

The `ScheduledTaskManager` provides a unified system for all periodic tasks:
- Content indexing (podcast, YouTube, Rumble)
- Embedding hydration
- Speaker identification
- Podcast chart collection
- Cache refresh (replaces pg_cron)

## Components

| File | Purpose |
|------|---------|
| `scheduled_task_manager.py` | Core scheduler - loads tasks from config, manages execution |
| `schedule_types.py` | Schedule type definitions (interval, time_of_day, run_then_wait) |
| `background_loops.py` | Background tasks (stats, config reload, cleanup, health) |
| `task_creation_manager.py` | Task creation logic for content pipeline |
| `executors/` | Task executors (CLI commands, SQL functions) |

## Task Modules

All scheduled tasks are implemented as Python modules with `__main__` entry points:

| Module | Purpose | Scheduled Task |
|--------|---------|----------------|
| `src.automation.task_creation_manager` | Index sources, create processing tasks | `podcast_index_download`, `youtube_index_download` |
| `src.utils.embedding_hydrator` | Generate embeddings for segments | `embedding_hydration` |
| `src.speaker_identification.orchestrator` | Speaker identification phases | `speaker_id_phase1`, `speaker_id_phase2` |
| `src.ingestion.podcast_pipeline` | Monthly podcast chart collection | `podcast_collection` |

## Configuration

Tasks are defined in `config/config.yaml` under `scheduled_tasks:`:

```yaml
scheduled_tasks:
  enabled: true
  state_file: "/path/to/state.json"

  tasks:
    podcast_index_download:
      name: "Podcast Index & Download"
      description: "Index podcast feeds and create download tasks"
      enabled: true
      schedule:
        type: time_of_day
        hours: [0, 8, 16]
      executor:
        type: cli
        command: "python"
        args: ["-m", "src.automation.task_creation_manager", "--steps", "index_podcast", "download_podcast"]
      timeout_seconds: 3600
```

## Schedule Types

- **interval**: Run every N seconds
- **time_of_day**: Run at specific hours (optionally with days_interval)
- **run_then_wait**: Run, then wait N seconds after completion

## Executor Types

- **cli**: Run Python modules via `-m` or shell commands
- **sql**: Execute PostgreSQL functions

## API

The orchestrator exposes these endpoints:

```
GET  /api/scheduled_tasks/status              # All tasks
GET  /api/scheduled_tasks/{task_id}/status    # Specific task
POST /api/scheduled_tasks/{task_id}/trigger   # Manual trigger
POST /api/scheduled_tasks/{task_id}/enable    # Enable/disable
```

## State Persistence

Task state (last run time, result) is persisted to JSON file and survives restarts.

## Running Modules Manually

```bash
# Task creation
python -m src.automation.task_creation_manager --steps index_podcast download_podcast
python -m src.automation.task_creation_manager --project CPRMV --steps download convert

# Embedding hydration
python -m src.utils.embedding_hydrator --batch-size 128

# Speaker identification
python -m src.speaker_identification.orchestrator --phases 1 --all-active --apply

# Podcast pipeline
python -m src.ingestion.podcast_pipeline --phase all
```
