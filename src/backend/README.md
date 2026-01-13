# Backend API Service

**Status:** Production-ready ✅ | **Architecture Grade:** B+ | **Version:** 1.0 (Simplified)

Simplified FastAPI backend exposing **3 core endpoints** for media and analysis operations.

**Recent Updates (November 2025):**
- ✅ Simplified from 9 routers to 3 focused endpoints
- ✅ Migrated from FAISS to pgvector (200-750x faster incremental refresh)
- ✅ Unified media + transcription endpoint
- ✅ Declarative pipeline system for all RAG operations
- ✅ PostgreSQL-based LLM caching (~$200-300/month savings)

## Architecture

This backend implements a **modular, 3-layer RAG (Retrieval-Augmented Generation) architecture** designed for composability, performance, and extensibility.

### Technology Stack

**API Layer:**
- **FastAPI** - Modern async Python web framework with SSE streaming
- **3 focused endpoints** - Health, media, analysis (down from 9 routers)

**Search & Retrieval:**
- **pgvector** - PostgreSQL vector search with IVFFlat indexes
- **Incremental refresh** - Automated updates via pg_cron (200-750x faster than full rebuilds)
- **Materialized cache tables** - 7d, 30d, 180d rolling windows

**LLM & Embeddings:**
- **Grok API** - Text generation (query expansion, summarization)
- **Qwen2-Instruct** - Multilingual embeddings (1024-dim, 2000-dim)
- **Pre-loaded models** - Avoid PyTorch/MPS initialization issues
- **PostgreSQL cache** - Semantic similarity matching (82% hit rate)

**Analysis:**
- **HDBSCAN + UMAP** - Theme clustering with adaptive validation
- **Quantitative analysis** - Volume, centrality, discourse metrics
- **Declarative pipelines** - Composable workflow steps

### Modular RAG Architecture

**Layer 1: Data Retrieval** (Core data access and indexing)
- `SegmentRetriever` - Unified segment fetching with flexible filters (database-first)
- `QueryParser` - LLM-based query expansion (multi_query, query2doc, theme_queries)
- `SmartRetriever` - High-level retrieval combining query parsing + semantic search

**Layer 2: Analysis Primitives** (Reusable analysis components)
- `ThemeExtractor` - Theme discovery via clustering (UMAP + HDBSCAN) with adaptive validation
- `SegmentSelector` - Weighted segment sampling (diversity, centrality, recency, quality)
- `TextGenerator` - LLM text generation with prompt templates and batch processing
- `QuantitativeAnalyzer` - Volume metrics, channel distribution, discourse centrality

**Layer 3: Workflows** (Composable analysis pipelines)
- `AnalysisPipeline` - Declarative workflow orchestration with step registry
- `SimpleRAGWorkflow` - Query → Retrieve → Sample → Summarize
- `HierarchicalSummaryWorkflow` - Multi-stage theme extraction and summarization
- Streaming support via Server-Sent Events (SSE) for progressive results

### Key Features
- ✅ **pgvector incremental refresh** - 200-750x faster than FAISS full rebuilds
- ✅ **Declarative pipelines** - Configure workflows via JSON (no code changes)
- ✅ **Parallel execution** - Async operations with rate limiting (20-50 concurrent LLM calls)
- ✅ **SSE streaming** - Progressive updates to frontend
- ✅ **Hierarchical analysis** - Main themes + sub-themes with adaptive clustering
- ✅ **Semantic LLM cache** - 52-82% hit rates, ~$200-300/month savings
- ✅ **Unified media endpoint** - Audio/video + optional AssemblyAI transcription

📖 **Documentation:**
- **[SSE Streaming Specification](SSE_STREAMING.md)** - Complete SSE event format and implementation guide
- **[RAG Infrastructure](app/services/rag/README.md)** - Pipeline architecture, available steps, and workflows
- **[Architecture Review](ARCHITECTURE_REVIEW.md)** - Complete architecture assessment and evaluation
- **[Cleanup Summary](CLEANUP_SUMMARY.md)** - Recent simplification and improvements
- **[Cache Analysis](CACHE_SERVICES_ANALYSIS.md)** - Cache effectiveness and cost savings
- **[Archive Index](archive/ARCHIVE_INDEX.md)** - Deprecated code and restoration instructions

## Setup

### 1. Install Dependencies

```bash
cd src/backend
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

### 3. Run Server

```bash
# Development mode (auto-reload)
python app/main.py

# Or with uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload

# Production mode in screen
screen -S backend
uvicorn app.main:app --host 0.0.0.0 --port 8002 --workers 1
# Ctrl+A, D to detach
```

## API Endpoints

The backend exposes **3 core endpoints** for all functionality:

### 1. Health & Monitoring
- `GET /health` - Service health check
- `GET /health/models` - Check if embedding models loaded
- `GET /health/db` - Check database connection

### 2. Media (`/api/media`)
**Unified media serving with optional transcription**

```bash
# Basic media retrieval
GET /api/media/content/{content_id}?start_time=10&end_time=30

# With transcription (batch mode)
GET /api/media/content/{content_id}?start_time=10&end_time=30&transcribe=true

# Streaming mode (SSE)
GET /api/media/content/{content_id}?start_time=10&end_time=30&transcribe=true&stream=true
```

**Features:**
- Audio/video extraction from S3 storage
- Automatic format detection (video → audio fallback)
- Optional AssemblyAI transcription with speaker labels
- Translation support (English + French)
- In-memory caching (500MB, 5min TTL)
- Two modes: Batch (JSON) or Streaming (SSE)

**Parameters:**
- `media_type`: `auto` (default), `video`, `audio`
- `format`: `mp4` (default), `webm`
- `transcribe`: Enable transcription (default: false)
- `stream`: Enable SSE streaming (default: false)
- `language`: Language code or auto-detect
- `speaker_labels`: Enable diarization (default: true)
- `enable_translation`: Translate to en/fr (default: true)

### 3. Analysis (`/api/analysis`)
**All RAG and search operations with declarative workflows and SSE streaming**

```bash
# Streaming mode (recommended) - Normal mode
POST /api/analysis/stream
{
  "query": "carbon tax",
  "dashboard_id": "cprmv-practitioner",
  "workflow": "simple_rag"
}

# Streaming mode - Verbose mode (debugging)
POST /api/analysis/stream
{
  "query": "carbon tax",
  "dashboard_id": "cprmv-practitioner",
  "workflow": "simple_rag",
  "verbose": true
}

# Batch mode
POST /api/analysis
{
  "query": "Pierre Poilievre",
  "dashboard_id": "cprmv-practitioner",
  "workflow": "simple_rag"
}

# Custom pipeline
POST /api/analysis/stream
{
  "query": "inflation",
  "dashboard_id": "cprmv-practitioner",
  "pipeline": [
    {"step": "expand_query", "config": {"strategy": "multi_query"}},
    {"step": "retrieve_segments", "config": {"k": 200}},
    {"step": "select_segments", "config": {"n": 20}},
    {"step": "generate_summary", "config": {}}
  ]
}
```

**Features:**
- **Step-based pipeline architecture** with declarative configuration
- **SSE streaming** with two modes:
  - **Normal mode** (`verbose=false`, default): Emits `result` + `complete` events only (6 events for 5-step pipeline)
  - **Verbose mode** (`verbose=true`): Adds `progress` + `partial` events for debugging (30+ events)
- **Result events**: Each pipeline step emits its own `result` event with step-specific output data
- Multiple workflows: `simple_rag`, `hierarchical_summary`, `hierarchical_with_subthemes`, `search_only`, `deep_analysis`
- Query expansion strategies: `multi_query`, `query2doc`, `theme_queries`, `stance_variation`
- pgvector semantic search with incremental refresh
- Theme extraction and sub-theme detection
- Quantitative analysis (volume, channels, discourse centrality)
- Flexible pipeline composition: Add custom steps and workflows

**SSE Event Flow:**
1. Each step emits a `result` event with its output data when complete
2. Frontend collects data from `result` events (e.g., `quantitative_metrics`, `selected_segments`, `summaries`)
3. Final `complete` event signals pipeline completion (contains no data, just timing/status)

**Discovery endpoints:**
- `GET /api/analysis/steps` - List available pipeline steps
- `GET /api/analysis/workflows` - List predefined workflows

📖 **See [app/services/rag/README.md](app/services/rag/README.md) for detailed pipeline architecture and step definitions**

## Database Infrastructure

### pgvector Embedding Cache Tables (Incremental Refresh)

The backend uses PostgreSQL tables with incremental refresh for embedding search. Unlike traditional materialized views that rebuild everything, these tables update only changed data via **pg_cron** scheduled jobs.

#### Architecture

**Regular tables with incremental updates:**
- All tables (180d, 30d, 7d) pull directly from `embedding_segments` + `content` (source of truth)
- No cascading dependencies - each window operates independently
- Uses `ON CONFLICT` upserts to handle overlapping lookback windows
- Primary key on `id` prevents duplicates

**Performance improvement over materialized views:**
- 180d daily refresh: ~2K rows vs 1.5M rows (**750x faster**)
- 30d 6h refresh: ~500 rows vs 250K rows (**500x faster**)
- 7d hourly refresh: ~200 rows vs 40K rows (**200x faster**)

#### Active Scheduled Jobs

| Job Name | Schedule | Purpose | Performance | Status |
|----------|----------|---------|-------------|--------|
| `refresh-180d-main-incr` | Daily at 2:00 AM | Incremental refresh 180-day main cache | ~2K rows | ✅ Active |
| `refresh-180d-alt-incr` | Daily at 2:10 AM | Incremental refresh 180-day alt cache | ~2K rows | ✅ Active |
| `refresh-30d-main-incr` | Every 6h at :30 | Incremental refresh 30-day main cache | ~500 rows | ✅ Active |
| `refresh-30d-alt-incr` | Every 6h at :40 | Incremental refresh 30-day alt cache | ~500 rows | ✅ Active |
| `refresh-7d-main-incr` | Hourly at :50 | Incremental refresh 7-day hot cache | ~200 rows | ✅ Active |
| `refresh-7d-alt-incr` | Hourly at :55 | Incremental refresh 7-day hot cache | ~200 rows | ✅ Active |
| `reconcile-180d-main-weekly` | Sunday 3:00 AM | Full rebuild to catch drift | Full scan | ✅ Active |
| `reconcile-180d-alt-weekly` | Sunday 3:30 AM | Full rebuild to catch drift | Full scan | ✅ Active |
| `cleanup-llm-cache` | Daily at 1:00 AM | Remove expired LLM cache entries | - | ✅ Active |

#### Cache Tables

- **`embedding_cache_180d`** - Main embedding model, 180-day rolling window (~1.5M rows)
- **`embedding_cache_180d_alt`** - Alt embedding model, 180-day rolling window (~145K rows)
- **`embedding_cache_30d`** - 30-day rolling window for medium-term queries (~270K rows)
- **`embedding_cache_30d_alt`** - Alt embedding model, 30-day window (~7K rows)
- **`embedding_cache_7d`** - Hot cache for recent content (main model) (~18K rows)
- **`embedding_cache_7d_alt`** - Hot cache for recent content (alt model) (~300 rows)

Each table includes:
- Primary key on `id` (from `embedding_segments.id`)
- GIN index on `projects` array for project filtering
- B-tree indexes on `publish_date`, `content_id` for efficient filtering
- `cache_refreshed_at` timestamp for tracking freshness

#### Sync Guarantees

The incremental refresh system prevents data drift via:

1. **Captures all changes:** Tracks `last_updated` timestamps, catches INSERTs, UPDATEs, metadata changes
2. **Handles deletions:** Removes rows outside time window and orphaned rows (deleted from source)
3. **Prevents drift:** Weekly full reconciliation rebuilds from source to catch any issues
4. **Idempotent operations:** `ON CONFLICT DO UPDATE` ensures no duplicates
5. **Time buffers:** Lookback windows have 1-hour buffers to catch late-arriving data

#### Monitoring

```sql
-- Check cache freshness and health
SELECT * FROM embedding_cache_health ORDER BY cache_name;

-- Check for drift between cache and source
SELECT * FROM embedding_cache_drift_check;

-- View all scheduled jobs
SELECT jobid, jobname, schedule, active, command
FROM cron.job
WHERE jobname LIKE '%refresh%' OR jobname LIKE '%reconcile%'
ORDER BY jobname;

-- Check recent job execution
SELECT jobid, runid, status, start_time, end_time, return_message
FROM cron.job_run_details
WHERE jobid IN (SELECT jobid FROM cron.job WHERE jobname LIKE '%refresh%')
ORDER BY start_time DESC LIMIT 10;
```

**Migration files:**
- `migrations/create_embedding_cache_views.sql` (deprecated - materialized views)
- `migrations/robust_incremental_refresh.sql` (current - incremental tables)

## Documentation

- Interactive API docs: http://localhost:8002/docs
- ReDoc: http://localhost:8002/redoc

## Development

### Running Tests

```bash
# Run all active tests
pytest tests/ -v

# Run specific test
pytest tests/test_analysis_pipeline.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

**Active tests (8 files):**
- `test_analysis_pipeline.py` - Pipeline orchestration
- `test_layer1.py` - Data retrieval components
- `test_query_parser.py` - Query expansion
- `test_segment_selector.py` - Segment sampling
- `test_text_generator.py` - LLM generation
- `test_theme_extractor.py` - Theme clustering
- `test_simple_rag_workflow.py` - SimpleRAG workflow
- `test_simple_rag_integration.py` - End-to-end integration

**Note:** Test isolation files and adhoc tests have been archived to `archive/test_isolation/` and `archive/tests_adhoc/`

### Logs

```bash
# Backend logs
tail -f logs/backend.log

# Usage/activity logs
tail -f logs/activity.log
```

### Testing Zone

**Organized testing structure:**
- **`tests/`** - 8 unit & integration tests (pytest)
- **`testing_utils/`** - SSE client, validators, test runner, report generation
- **`utilities/`** - Analysis scripts and debug utilities (5 files)

See **[TESTING_GUIDE.md](TESTING_GUIDE.md)** for comprehensive testing documentation.

---

## Architecture & Cleanup

### Recent Improvements (November 2025)

**API Simplification:**
- ✅ Reduced from 9 routers to 3 focused endpoints (67% reduction)
- ✅ Merged transcription into unified media endpoint
- ✅ Consolidated all RAG/search into analysis endpoint
- ✅ Internalized LLM/embeddings services (not exposed as direct APIs)

**Services Cleanup:**
- ✅ Migrated from FAISS to pgvector (200-750x faster incremental refresh)
- ✅ Replaced file-based cache with PostgreSQL (concurrent-safe, queryable)
- ✅ Archived 4 deprecated services (~50KB code)

**Models Cleanup:**
- ✅ Archived 17+ deprecated Pydantic models
- ✅ Documented all deprecations with clear replacements

**Tests Cleanup:**
- ✅ Archived 18 debug/adhoc test files
- ✅ Kept 8 active unit/integration tests

### Documentation

**Comprehensive reviews:**
- `ARCHITECTURE_REVIEW.md` - Full architecture assessment
- `CLEANUP_SUMMARY.md` - Complete cleanup actions and metrics
- `CACHE_SERVICES_ANALYSIS.md` - Cache effectiveness analysis
- `FINAL_CLEANUP_REPORT.md` - Final status and recommendations
- `archive/ARCHIVE_INDEX.md` - Guide to archived code

### Architecture Grade: **B+**

**Strengths:**
- Clean API surface (3 endpoints)
- Single source of truth for each capability
- Well-documented and maintainable
- Production-ready with good performance

**Path to A:**
- Remove dead imports (5 min)
- Review scripts directory (30 min)
- Consider refactoring llm_service.py (optional)
