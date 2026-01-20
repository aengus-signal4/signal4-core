# Processing Repository

Content ingestion, processing pipeline, and query API. **Owns the database schema.**

**Repository:** https://github.com/aengus-signal4/signal4-core

## Quick Reference

- **Orchestrator**: Task Orchestrator V2 (port 8001)
- **Workers**: Task Processor (port 8000 on each worker)
- **Backend API**: Query & Analysis API (port 7999)
- **Database**: PostgreSQL at 10.0.0.4:5432 (READ/WRITE for processing, READ-ONLY for backend)
- **Storage**: MinIO at 10.0.0.251:9000

## Directory Structure

```
core/
├── src/
│   ├── orchestration/                # Task orchestration system
│   │   ├── orchestrator.py           # TaskOrchestratorV2 (main entry, port 8001)
│   │   ├── api_endpoints.py          # REST API for orchestrator
│   │   ├── task_manager.py           # Task queue management
│   │   ├── config_manager.py         # Config loading with hot-reload
│   │   ├── reactive_assignment.py    # Dynamic task assignment
│   │   ├── timeout_manager.py        # Task timeout detection
│   │   ├── failure_tracker.py        # Worker failure tracking
│   │   └── logging_setup.py          # Orchestrator logging
│   │
│   ├── automation/                   # Scheduled task system
│   │   ├── scheduled_task_manager.py # Unified scheduler (config-driven)
│   │   ├── schedule_types.py         # Schedule definitions
│   │   ├── background_loops.py       # Stats, config reload, cleanup
│   │   └── executors/                # CLI and SQL task executors
│   │
│   ├── workers/                      # Worker management
│   │   ├── processor.py              # Task processor (port 8000)
│   │   ├── pool.py                   # Worker pool management
│   │   ├── service_manager.py        # Worker service lifecycle
│   │   ├── service_startup.py        # LLM server + dashboards startup
│   │   └── management.py             # Worker SSH/process management
│   │
│   ├── monitoring/                   # Health monitoring
│   │   ├── network.py                # Worker network health
│   │   ├── network_monitor.py        # Detailed network monitoring
│   │   └── s3.py                     # S3/MinIO health monitoring
│   │
│   ├── processing_steps/             # Pipeline stages
│   │   ├── download_youtube.py       # YouTube downloads (yt-dlp)
│   │   ├── download_podcast.py       # Podcast downloads
│   │   ├── download_rumble.py        # Rumble downloads
│   │   ├── convert.py                # Audio conversion, VAD, chunking
│   │   ├── transcribe.py             # Speech-to-text (Whisper/MLX)
│   │   ├── diarize.py                # Speaker diarization (Pyannote)
│   │   ├── stitch.py                 # Combine chunks into final transcript
│   │   ├── segment_embeddings.py     # Generate embeddings for segments
│   │   ├── cleanup_and_compress.py   # Archive and compress
│   │   └── stitch_steps/             # 14 sub-steps for stitching
│   │
│   ├── services/                     # Shared services
│   │   └── llm/                      # LLM infrastructure
│   │       ├── balancer.py           # LLM load balancer (port 8002)
│   │       ├── mlx_server.py         # MLX model server (port 8004)
│   │       └── model_config.py       # Model requirements per task
│   │
│   ├── ingestion/                    # Content discovery
│   │   ├── youtube_indexer.py        # YouTube channel/playlist indexing
│   │   ├── podcast_indexer.py        # Podcast RSS feed indexing
│   │   ├── rumble_indexer.py         # Rumble channel indexing
│   │   └── podcast_enricher.py       # Metadata enrichment
│   │
│   ├── speaker_identification/       # Speaker ID pipeline
│   │   ├── orchestrator.py           # Speaker ID orchestration
│   │   └── strategies/               # ID strategies (merge, evidence, etc.)
│   │
│   ├── classification/               # Content classification
│   │   └── semantic_theme_classifier.py
│   │
│   ├── database/                     # DATABASE SCHEMA OWNER
│   │   ├── models.py                 # All SQLAlchemy models (source of truth)
│   │   ├── session.py                # Database session management
│   │   ├── manager.py                # High-level DB operations
│   │   └── state_manager.py          # Content state tracking
│   │
│   ├── storage/                      # S3/MinIO utilities
│   │   ├── s3_utils.py               # S3 operations with failover
│   │   └── content_storage.py        # Content path management
│   │
│   ├── utils/                        # Shared utilities
│   │   ├── config.py                 # Config loading with env substitution
│   │   ├── node_utils.py             # Node detection (is_head_node, get_worker_name)
│   │   ├── paths.py                  # Path utilities (get_project_root)
│   │   ├── ip_utils.py               # IP/network utilities
│   │   ├── logger.py                 # Centralized logging
│   │   ├── llm_client.py             # Unified LLM client
│   │   ├── human_behavior.py         # Download rate limiting
│   │   └── embedding_hydrator.py     # Batch embedding generation
│   │
│   └── backend/                      # Query & Analysis API (port 7999)
│       ├── app/
│       │   ├── main.py               # FastAPI entry point
│       │   ├── routers/              # API endpoints
│       │   │   ├── health.py         # Health checks
│       │   │   ├── query.py          # Content queries
│       │   │   ├── analysis.py       # RAG analysis (SSE streaming)
│       │   │   └── media.py          # Media streaming
│       │   ├── services/             # Business logic
│       │   │   ├── embedding_service.py
│       │   │   ├── pgvector_search_service.py
│       │   │   ├── llm_service.py
│       │   │   └── rag/              # RAG pipeline components
│       │   ├── models/               # Request/Response DTOs
│       │   └── config/               # Dashboard configurations
│       ├── scripts/                  # Backend operational scripts
│       │   ├── audit_cache_and_clustering.py
│       │   └── generate_embedding_clusters.py
│       └── run_server.py
│
├── dashboards/                       # OPERATIONAL dashboards only
│   ├── orchestrator_monitoring.py    # Task queue, worker status (port 8503)
│   ├── worker_monitoring_v2.py       # Worker health, task assignment (port 8501)
│   ├── project_monitoring.py         # Project progress (port 8502)
│   └── system_monitoring.py          # System health
│
├── migrations/                       # Alembic migrations (schema changes)
│   ├── alembic.ini
│   └── versions/
│
├── scripts/                          # Operational scripts
│   ├── create_tasks_db.py            # Task creation script
│   ├── hydrate_embeddings.py         # Embedding generation
│   └── collect_and_classify_podcasts.py
│
├── config/
│   └── config.yaml                   # Main configuration (scheduled_tasks, workers, etc.)
│
├── .env                              # Credentials (not in git)
├── .env.example                      # Credential template
├── pyproject.toml                    # Python dependencies (managed by uv)
├── uv.lock                           # Locked dependency versions
└── README.md
```

## Environment Setup

We use [uv](https://docs.astral.sh/uv/) for fast, deterministic dependency management.

```bash
# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync environment (run after git pull)
uv sync

# Or install with all optional dependencies
uv sync --all-extras
```

## Key Commands

```bash
# Start orchestrator (V2)
uv run python -m src.orchestration.orchestrator

# Start worker (on worker machines)
uv run python -m uvicorn src.workers.processor:app --host 0.0.0.0 --port 8000

# Start LLM balancer
uv run python -m uvicorn src.services.llm.balancer:app --host 0.0.0.0 --port 8002

# Start backend API
uv run python -m uvicorn src.backend.app.main:app --host 0.0.0.0 --port 7999

# Run database migrations
uv run alembic upgrade head

# Add a new dependency
uv add <package>

# Update lockfile after pyproject.toml changes
uv lock
```

## Scheduled Tasks

All periodic tasks are managed by `ScheduledTaskManager` via config.yaml under `scheduled_tasks:`.

**Task types:**
- `podcast_index_download` - Index podcast feeds & create download tasks
- `youtube_index_download` - Index YouTube channels & create download tasks
- `embedding_hydration` - Generate embeddings for segments
- `speaker_id_phase1/2` - Speaker identification phases
- `podcast_collection` - Monthly podcast chart collection
- `cache_refresh_*` - Cache refresh jobs (replaces pg_cron)

**API endpoints:**
- `GET /api/scheduled_tasks/status` - All tasks status
- `GET /api/scheduled_tasks/{task_id}/status` - Specific task status
- `POST /api/scheduled_tasks/{task_id}/trigger` - Manual trigger

## Backend API

The backend provides read-only query and analysis APIs for the frontend. Located at `src/backend/`.

**Key endpoints:**
- `POST /api/query` - Query content with filters
- `POST /api/search` - Semantic search via pgvector
- `POST /api/analysis/stream` - RAG analysis with SSE streaming
- `GET /api/media/content/{id}` - Stream audio/video

**Important:** Backend should NEVER write to the database (except LLM cache).

## Backend Logging

Hybrid logging: workflow milestones → console + `logs/backend/workflow.log`, component debug → per-component files (no console). See `.claude/docs/backend-logging.md` for details.

## Pipeline Stages

```
download -> convert -> transcribe -> diarize -> stitch -> segment -> cleanup
```

Each stage creates a task in the queue. Workers pick up tasks based on their configured capabilities.

## Configuration

See `config/config.yaml` for:
- `scheduled_tasks:` - Periodic task definitions
- `processing.workers:` - Worker definitions (IPs, task types, limits)
- `storage.s3:` - S3 endpoints and bucket
- `database:` - Database connection
- `processing.llm_server:` - LLM server configuration

**Credentials:** All secrets are in `.env` (copied from `.env.example`):
- `POSTGRES_PASSWORD`
- `S3_ACCESS_KEY`, `S3_SECRET_KEY`
- `HF_TOKEN` (HuggingFace)

Config values can reference env vars using `${VAR}` syntax (e.g., `password: "${POSTGRES_PASSWORD}"`).

**Loading config in code:**
```python
from src.utils.config import load_config, get_credential

config = load_config()  # Loads YAML with ${VAR} substitution
token = get_credential('HF_TOKEN')  # Direct credential access
```

## Database Schema

This repo owns the schema. Key models in `src/database/models.py`:
- `Content` - Media items (videos, podcasts)
- `ContentChunk` - Audio chunks for processing
- `Speaker` - Identified speakers
- `SpeakerTranscription` - Speaker-attributed text
- `EmbeddingSegment` - Text segments with embeddings
- `TaskQueue` - Processing task queue

When changing schema:
1. Edit `src/database/models.py`
2. Create migration: `alembic revision --autogenerate -m "description"`
3. Apply: `alembic upgrade head`

## Worker Network

Workers are Mac machines on local network:
- Head node: 10.0.0.4 (runs orchestrator)
- worker0: 10.0.0.34 (MLX tier-1)
- worker1-5: Various IPs
- Each has eth (primary) and wifi (fallback) interfaces

## Quick Investigation Commands

Use these commands to quickly investigate content issues. Environment variables are loaded from `.env`.

### PostgreSQL Access

```bash
# Quick query (loads creds from .env automatically)
cd ~/signal4/core
source .env && psql -h 10.0.0.4 -U signal4 -d av_content

# One-liner queries
source .env && psql -h 10.0.0.4 -U signal4 -d av_content -c "SELECT * FROM content WHERE content_id = 'pod_9bb4e73a66d8'"

# Check content and chunks for an item
source .env && psql -h 10.0.0.4 -U signal4 -d av_content -c "
SELECT c.content_id, c.title, c.is_transcribed, c.processing_stage,
       COUNT(ch.id) as chunks,
       SUM(CASE WHEN ch.transcription_status = 'completed' THEN 1 ELSE 0 END) as completed
FROM content c
LEFT JOIN content_chunks ch ON c.content_id = ch.content_id
WHERE c.content_id = 'CONTENT_ID_HERE'
GROUP BY c.content_id, c.title, c.is_transcribed, c.processing_stage"

# Check chunk statuses for a content item
source .env && psql -h 10.0.0.4 -U signal4 -d av_content -c "
SELECT chunk_index, transcription_status, transcription_model, diarization_status
FROM content_chunks WHERE content_id = 'CONTENT_ID_HERE' ORDER BY chunk_index"
```

### S3/MinIO Access

```bash
# Configure mc alias (one-time setup)
source .env && mc alias set minio http://10.0.0.251:9000 $S3_ACCESS_KEY $S3_SECRET_KEY

# List files for a content item
mc ls minio/av-content/content/CONTENT_ID_HERE/

# List chunk files
mc ls minio/av-content/content/CONTENT_ID_HERE/chunks/

# Check specific chunk
mc ls minio/av-content/content/CONTENT_ID_HERE/chunks/11/

# View transcript file
mc cat minio/av-content/content/CONTENT_ID_HERE/chunks/11/transcript_words.json | jq .

# Check if file exists (returns exit code)
mc stat minio/av-content/content/CONTENT_ID_HERE/chunks/11/transcript_words.json

# Download file for inspection
mc cp minio/av-content/content/CONTENT_ID_HERE/chunks/11/transcript_words.json /tmp/
```

### Using Python Utils

```python
# Quick database session
from src.database.session import get_session
from src.database.models import Content, ContentChunk

with get_session() as session:
    content = session.query(Content).filter_by(content_id='pod_9bb4e73a66d8').first()
    print(f"Title: {content.title}, Stage: {content.processing_stage}")

    chunks = session.query(ContentChunk).filter_by(content_id='pod_9bb4e73a66d8').all()
    for c in chunks:
        print(f"Chunk {c.chunk_index}: transcription={c.transcription_status}")

# Quick S3 access
from src.storage.s3_utils import S3Storage, S3StorageConfig
from src.utils.config import load_config

config = load_config()
s3_config = S3StorageConfig.from_dict(config['storage']['s3'])
s3 = S3Storage(s3_config)

# List files
files = s3.list_files('content/pod_9bb4e73a66d8/chunks/11/')
print(files)

# Check if file exists
exists = s3.file_exists('content/pod_9bb4e73a66d8/chunks/11/transcript_words.json')

# Download file content
content = s3.download_json('content/pod_9bb4e73a66d8/chunks/11/transcript_words.json')
```

## Development Norms
- **DO NOT test** until specifically asked
- **Always remove old code** - no backward compatibility needed
- Check for existing outputs before processing
- Update `content.last_updated` when modifying state
- Use structured logging and proper error handling

## Documentation
- Keep journal in `journal/yyyy-mm-dd.md`
- Backend API & RAG: `src/backend/README.md` and related docs
- Config: `config/config.yaml` (projects, worker IDs, models, storage, database)
