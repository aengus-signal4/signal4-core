# Signal4 Processing

Content ingestion and processing pipeline for audio-visual media.

## Overview

This repository handles:
- Content discovery and downloading (YouTube, podcasts, Rumble)
- Audio processing (conversion, VAD, chunking)
- Speech-to-text transcription (Whisper/MLX)
- Speaker diarization (Pyannote)
- Transcript stitching and LLM processing
- Embedding generation for semantic search

## Quick Start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies (locked versions)
uv sync

# Or install with all optional dependencies (ML, NLP, tools)
uv sync --all-extras

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run database migrations
uv run alembic upgrade head

# Start orchestrator
uv run python -m src.orchestration.orchestrator

# Start worker (on worker machines)
uv run python -m uvicorn src.workers.processor:app --host 0.0.0.0 --port 8000
```

## Environment Management with uv

We use [uv](https://docs.astral.sh/uv/) for fast, deterministic dependency management.

**Key commands:**

```bash
# Sync environment to match lockfile (run on each machine)
uv sync

# Add a new dependency
uv add <package>

# Add to optional group (ml, nlp, or tools)
uv add <package> --optional ml

# Update lockfile after pyproject.toml changes
uv lock

# Run any command in the virtual environment
uv run <command>
```

**Files:**
- `pyproject.toml` - Dependencies and project config
- `uv.lock` - Locked versions (commit this!)
- `.venv/` - Virtual environment (gitignored)

**Updating dependencies on worker machines:**
```bash
cd ~/signal4/core
git pull
uv sync
```

## Worker Setup

See [docs/worker_setup.md](docs/worker_setup.md) for worker machine setup and troubleshooting.

## Architecture

See [CLAUDE.md](CLAUDE.md) for detailed architecture documentation.

## Configuration

Edit `config/config.yaml` to configure:
- Worker machines and their capabilities
- Task type assignments
- S3 storage endpoints
- Database connection

## Pipeline

```
download -> convert -> transcribe -> diarize -> stitch -> embed
```

## License

Proprietary - Signal4
