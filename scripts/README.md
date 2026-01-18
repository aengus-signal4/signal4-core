# Scripts Directory

Utility shell scripts for deployment and worker management.

## Worker Setup

### Prerequisites

Each worker requires:
- **Homebrew** - Package manager for macOS
- **ffmpeg** - Audio/video processing (via Homebrew)
- **uv** - Python package manager

### Initial Setup

```bash
# On each worker, install ffmpeg via Homebrew
brew install ffmpeg

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or use the setup script from the head node:
```bash
./scripts/install_uv_workers.sh        # All workers
./scripts/install_uv_workers.sh 3,4    # Specific workers
```

### Important: Remove Conda from PATH

If conda was previously used, remove it from `~/.zshrc`:
```bash
# Remove these lines if present:
# source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
# conda activate content-processing
```

Use the fix script to automatically clean this up:
```bash
./scripts/fix_worker_paths.sh          # All workers
./scripts/fix_worker_paths.sh 6        # Specific worker
```

## Shell Scripts

| Script | Purpose |
|--------|---------|
| `sync_workers.sh` | Sync code to worker machines |
| `install_uv_workers.sh` | Install uv and ffmpeg on worker machines |
| `fix_worker_paths.sh` | Remove conda from PATH and verify ffmpeg |

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
uv run python -m src.automation.task_creation_manager --steps index_podcast download_podcast
uv run python -m src.automation.task_creation_manager --project CPRMV --steps download convert

# Embedding hydration
uv run python -m src.utils.embedding_hydrator --batch-size 128

# Podcast pipeline
uv run python -m src.ingestion.podcast_pipeline --phase all
```
