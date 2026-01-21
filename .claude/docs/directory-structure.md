# Core Repository Directory Structure

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
│   │   ├── models/                   # SQLAlchemy models (source of truth)
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
