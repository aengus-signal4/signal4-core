# Processing Repository

Content ingestion, processing pipeline, and query API. **Owns the database schema.**

## Quick Reference

| Service | Port | Location |
|---------|------|----------|
| Orchestrator | 8001 | `src/orchestration/orchestrator.py` |
| Workers | 8000 | `src/workers/processor.py` |
| Backend API | 7999 | `src/backend/app/main.py` |
| Database | 5432 | PostgreSQL at 10.0.0.4 |
| Storage | 9000 | MinIO at 10.0.0.251 |

## Key Commands

```bash
uv run python -m src.orchestration.orchestrator     # Start orchestrator
uv run alembic upgrade head                         # Run migrations
uv run python -m uvicorn src.backend.app.main:app --port 7999  # Backend API
```

## Pipeline

```
download -> convert -> transcribe -> diarize -> stitch -> segment -> cleanup
```

## Database Schema (IMPORTANT)

### Content ID Types

The `Content` table has TWO different ID columns - common source of JOIN errors:

| Column | Type | Purpose | Example |
|--------|------|---------|---------|
| `id` | Integer | Primary key, used for FKs | `12345` |
| `content_id` | String | Business identifier | `"pod_9bb4e73a66d8"` |

**Foreign keys reference `content.id` (integer), NOT `content.content_id` (string).**

```sql
-- CORRECT: Join on integer PK
SELECT * FROM content c
JOIN embedding_segments es ON c.id = es.content_id

-- WRONG: Type mismatch error (varchar vs integer)
SELECT * FROM content c
JOIN embedding_segments es ON c.content_id = es.content_id
```

### Key Models (`src/database/models/`)

- `Content` - Media items (videos, podcasts)
- `Sentence` - Atomic transcript unit (primary)
- `EmbeddingSegment` - Text segments with embeddings
- `Speaker` - Identified speakers
- `TaskQueue` - Processing task queue

## Configuration

- **Config**: `config/config.yaml` (scheduled_tasks, workers, storage, database)
- **Credentials**: `~/signal4/.env` (POSTGRES_PASSWORD, S3_ACCESS_KEY, HF_TOKEN)
- **Loading**: `from src.utils.config import load_config, get_credential`

## Development Norms

- **DO NOT test** until specifically asked
- **Always remove old code** - no backward compatibility
- Backend should NEVER write to database (except LLM cache)

## Reference Documentation

For detailed information, read these files when needed:

| Topic | File |
|-------|------|
| Directory structure | `.claude/docs/directory-structure.md` |
| Investigation commands (psql, S3, Python) | `.claude/docs/investigation-commands.md` |
| Backend logging | `.claude/docs/backend-logging.md` |
