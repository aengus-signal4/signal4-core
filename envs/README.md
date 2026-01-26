# UV Sub-Environments for Signal4 Processors

Lightweight, task-specific virtual environments that reduce processor startup time by 40-75%.

## Overview

Each sub-environment contains only the dependencies required for its specific task, avoiding the overhead of loading the full 2.5GB monolithic environment.

## Startup Time Improvements

| Task Type | Full Env | Sub-Env | Savings |
|-----------|----------|---------|---------|
| cleanup | 2.10s | 0.85s | 60% faster |
| download | 1.70s | 0.45s | 74% faster |
| transcribe | 2.10s | 1.09s | 48% faster |
| diarize | 10.68s | 4.41s | 59% faster |

## Environment Sizes

| Environment | Size | Purpose |
|-------------|------|---------|
| Main (.venv) | 2.5G | Full environment |
| cleanup | 201M | Storage optimization, compression |
| convert | 171M | Audio/video conversion |
| download | 213M | YouTube, podcasts, Rumble |
| transcribe | 1.2G | Whisper, Parakeet, MLX |
| diarize | 922M | PyAnnote speaker diarization |
| stitch | 1.1G | Speaker attribution, segmentation |
| mlx-server | 698M | MLX LLM model server |

## Usage

### Running a processor with its sub-environment

```bash
# From the core directory
cd /Users/signal4/signal4/core

# Run download processor
uv run --project envs/download python -c "
import sys
sys.path.insert(0, '.')
from src.processing_steps.download_youtube import YouTubeDownloader
# ... your code
"

# Run cleanup processor
uv run --project envs/cleanup python -c "
import sys
sys.path.insert(0, '.')
from src.processing_steps.cleanup_and_compress import StorageOptimizer
# ... your code
"
```

### For the worker processor

The worker needs to be updated to use sub-environments. The key change is in `processor.py`:

```python
# Current approach
cmd = [uv_cmd, 'run', '--project', str(project_root), 'python', str(script_path)]

# New approach with sub-environments
env_map = {
    'download_youtube': 'download',
    'download_podcast': 'download',
    'download_rumble': 'download',
    'convert': 'convert',
    'cleanup_and_compress': 'cleanup',
    'transcribe': 'transcribe',
    'diarize': 'diarize',
    'diarize_pyannote': 'diarize',
    'stitch': 'stitch',
}
script_name = script_path.stem
env_name = env_map.get(script_name)
env_path = project_root / 'envs' / env_name if env_name else project_root

cmd = [uv_cmd, 'run', '--project', str(env_path), 'python', str(script_path)]
```

Note: The script needs `sys.path.insert(0, project_root)` at the start to find the `src` module.

## Syncing Environments

```bash
# Sync all sub-environments
cd /Users/signal4/signal4/core/envs
for env in cleanup convert download transcribe diarize stitch mlx-server; do
  echo "Syncing $env..."
  cd "$env" && uv sync && cd ..
done

# Sync a single environment
cd /Users/signal4/signal4/core/envs/download
uv sync
```

## Key Design Decisions

1. **`package = false`**: Sub-environments don't build as packages - they just install dependencies
2. **`compile-bytecode = false`**: Disabled to avoid recompilation overhead on each run
3. **Common dependencies**: All environments include base deps (aiohttp, requests, boto3, sqlalchemy, pgvector, bcrypt, pandas, pydantic)
4. **Path setup**: Scripts must add project root to sys.path to import `src` modules

## References

- [uv Workspaces](https://docs.astral.sh/uv/concepts/projects/workspaces/)
- [uv Project Configuration](https://docs.astral.sh/uv/concepts/projects/config/)
- GitHub Issue: #58
