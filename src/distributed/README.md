# Distributed Processing System

This directory contains the distributed processing system for content analysis.

## Architecture

The distributed processing system consists of:

1. **Task Queue Manager** (`task_queue.py`)
   - Manages task distribution and worker coordination
   - Handles task claiming, completion, and error states
   - Implements worker heartbeat system

2. **Worker System** (in `scripts/run_workers.py`)
   - Manages worker processes and their lifecycle
   - Handles task processing and limits
   - Supports multiple worker types:
     - `transcribe`: Audio transcription
     - `download_youtube`: YouTube video downloads
     - `download_podcast`: Podcast episode downloads
     - `extract_audio`: Audio extraction from videos
     - `stitch`: Transcript stitching

## Configuration

Worker configuration is managed through `config.yaml`:

```yaml
processing:
  distributed:
    worker_limits:
      transcribe:
        tasks_per_run: 12
        cooldown: 60
      download_youtube:
        tasks_per_run: 50
        cooldown: 3600
      # ... other worker types
  workers:
    10.0.0.209:  # s4-head
      enabled: true
      enabled_tasks: ["download_youtube", "extract_audio", "transcribe"]
      max_concurrent_tasks: 1
```

## Usage

Workers can be managed using the following scripts:

1. Start workers:
```bash
python scripts/run_workers.py --worker-type download_youtube transcribe
```

2. Manage workers:
```bash
python scripts/manage_workers.py start  # Start all workers
python scripts/manage_workers.py stop   # Stop all workers
```

3. Service management:
```bash
python scripts/service_manager.py start  # Start as service
python scripts/service_manager.py stop   # Stop service
```

## Worker Types

### YouTube Downloader
- Downloads videos from YouTube
- Enforces time windows (7 AM - random end time between 9:30-10:30 PM)
- Manages daily download limits
- Implements natural delays between downloads

### Transcription Worker
- Processes audio files for transcription
- Uses MLX Whisper for efficient transcription
- Handles chunking and batching

### Audio Extractor
- Extracts audio from video files
- Optimizes audio for transcription

### Podcast Downloader
- Downloads podcast episodes
- Manages episode metadata

### Transcript Stitcher
- Combines chunked transcripts
- Maintains timing and speaker information

## Error Handling

The system implements robust error handling:
- Automatic task retries
- Worker health monitoring
- Graceful shutdown
- Resource cleanup
- Process recovery

## Monitoring

Workers can be monitored through:
- Log files in `logs/`
- Database task queue status
- Worker heartbeat system
- Process status monitoring

## Worker Setup

### 1. Initial Setup
Each worker needs to be set up with:
- Python environment (conda env: content-processing)
- Required packages
- Access to the codebase (synced from head node)

### 2. NAS Storage Setup

#### One-time Setup per Worker
Before running workers, you need to set up NAS credentials on each worker. This is a one-time setup that requires interactive access:

1. SSH into each worker:
```bash
ssh signal4@<worker-ip>
```

2. Store NAS credentials in the keychain (this requires interactive access):
```bash
cd ~/content_processing
source /opt/homebrew/Caskroom/miniforge/base/etc/profile.d/conda.sh
conda activate content-processing
python3 -c "from src.utils.nas_utils import store_credentials; success = store_credentials('YOUR_NAS_PASSWORD'); print('Credentials stored successfully' if success else 'Failed to store credentials')"
```

3. Test the mount:
```bash
python3 -c "from src.utils.nas_utils import ensure_nas_mounted; success, msg = ensure_nas_mounted(); print(msg)"
```

#### Automated Setup
After the one-time credential setup, you can use `setup_worker_nas.sh` to:
- Configure mount points
- Set up auto-mounting in .zshrc
- Remove any old LaunchAgents
- Test the mount

### 3. Worker Management

Workers are managed through several scripts:
- `scripts/run_workers.py`: Main script to start and manage workers
- `scripts/run_worker.py`: Individual worker process script
- `scripts/restart_worker.sh`: Script to restart specific workers
- `scripts/shutdown_workers.py`: Script to cleanly shut down workers

### Architecture

The distributed system consists of:
- `task_queue.py`: Manages the task queue in PostgreSQL
- `worker.py`: Core worker implementation
- `worker_manager.py`: Manages worker lifecycle and health checks

### Task Types

Currently supported task types:
- `transcribe`: Audio transcription tasks
- `download_youtube`: YouTube video downloads
- `download_podcast`: Podcast episode downloads
- `extract_audio`: Audio extraction from videos

### Health Monitoring

Workers implement several health checks:
- Memory usage monitoring
- Task timeouts
- Error handling and recovery
- Automatic restarts on failure

### Troubleshooting

Common issues and solutions:

1. NAS Mount Issues
```bash
# Check mount status
python3 -c "from src.utils.nas_utils import check_mount_status; print(check_mount_status()[1])"

# Force remount
python3 -c "from src.utils.nas_utils import ensure_nas_mounted; print(ensure_nas_mounted()[1])"
```

2. Worker Process Issues
```bash
# Check worker processes
ps aux | grep run_worker

# View worker logs
tail -f /tmp/worker.log
```

3. Task Queue Issues
```bash
# Check task queue status
python3 scripts/check_queue.py

# Clear stuck tasks
python3 scripts/cleanup_tasks.py
``` 