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
| `executors/` | Task executors (CLI commands, SQL functions) |

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
        args: ["scripts/create_tasks_db.py", "--steps", "index_podcast"]
      timeout_seconds: 3600
```

## Schedule Types

- **interval**: Run every N seconds
- **time_of_day**: Run at specific hours (optionally with days_interval)
- **run_then_wait**: Run, then wait N seconds after completion

## Executor Types

- **cli**: Run shell commands (python scripts, etc.)
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
