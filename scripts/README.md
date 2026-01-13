# Scripts Directory

This directory is intentionally empty.

Previous operational scripts have been moved to their proper locations in `src/`:

- `hydrate_embeddings.py` → `src/utils/embedding_hydrator.py` (called by EmbeddingHydrationManager)
- `audit_content.py` → `src/processing/audit.py`
- `create_tasks_db.py` → Logic consolidated into `src/api/components/task_creation_manager.py`

All scripts are now integrated into the orchestrator and run automatically.
