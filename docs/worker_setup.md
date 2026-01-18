# Worker Setup Guide

Instructions for setting up and maintaining worker machines in the Signal4 processing cluster.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- SSH access configured from head node
- Network access to head node (10.0.0.4) and MinIO (10.0.0.251)

## Initial Setup

### 1. Clone Repository

```bash
cd ~
git clone <repo-url> signal4
cd signal4/core
```

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Install Dependencies

```bash
cd ~/signal4/core
uv sync --all-extras
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env with worker-specific settings
```

### 5. Create Log Directory

```bash
mkdir -p ~/logs/content_processing/$(hostname)
```

### 6. Start Worker

```bash
uv run python -m uvicorn src.workers.processor:app --host 0.0.0.0 --port 8000
```

## Required Python Packages

The following packages must be installed for full functionality. These should be in `pyproject.toml`, but verify they're present:

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| `aiofiles` | Async file I/O for podcast downloads | `uv add aiofiles` |
| `aiohttp` | Async HTTP client | `uv add aiohttp` |
| `yt-dlp` | Media downloading (YouTube, podcasts) | `uv add yt-dlp` |
| `ffmpeg-python` | Audio processing | `uv add ffmpeg-python` |

### Verify Installation

```bash
# Check a specific package
uv pip show aiofiles

# Check all critical packages
for pkg in aiofiles aiohttp yt-dlp ffmpeg-python; do
  uv pip show $pkg > /dev/null 2>&1 && echo "$pkg: OK" || echo "$pkg: MISSING"
done
```

## Updating Workers

After code changes on the head node:

```bash
cd ~/signal4/core
git pull
uv sync
```

To install a missing package across all workers:

```bash
# From head node
for worker in worker0 worker1 worker2 worker3 worker4 worker5; do
  ssh $worker "cd ~/signal4/core && uv add <package>"
done
```

## Troubleshooting

### Podcast Downloads Failing with "No module named 'aiofiles'"

The direct HTTP download method requires `aiofiles`. Without it, downloads fall back to the slower yt-dlp method.

**Fix:**
```bash
cd ~/signal4/core
uv add aiofiles
```

### Worker Not Responding

Check if the worker process is running:
```bash
ssh workerN "pgrep -f 'src.workers.processor'"
```

Check logs:
```bash
ssh workerN "tail -100 ~/logs/content_processing/workerN/workerN_task_processor.log"
```

### S3 Connection Issues

Verify MinIO is reachable:
```bash
curl -I http://10.0.0.251:9000/minio/health/live
```

## Worker Health Check

Run from head node to check all workers:

```bash
for worker in worker0 worker1 worker2 worker3 worker4 worker5; do
  echo "=== $worker ==="
  ssh -o ConnectTimeout=5 $worker "echo 'SSH: OK'" 2>/dev/null || echo "SSH: FAILED"
done
```
