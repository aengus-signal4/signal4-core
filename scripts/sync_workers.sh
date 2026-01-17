#!/bin/bash
# Sync code and dependencies to all workers

WORKERS="10.0.0.34 10.0.0.9 10.0.0.203 10.0.0.51 10.0.0.209"
SOURCE_DIR="/Users/signal4/signal4/core/"

for ip in $WORKERS; do
  echo "Syncing code to $ip..."
  rsync -av --delete \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='.env' \
    -e 'ssh -o StrictHostKeyChecking=no' \
    "$SOURCE_DIR" "signal4@$ip:/Users/signal4/signal4/core/" &
done
wait
echo "Code sync complete"

echo ""
echo "Running uv sync on all workers..."
for ip in $WORKERS; do
  echo "uv sync on $ip..."
  ssh -o StrictHostKeyChecking=no signal4@$ip 'cd /Users/signal4/signal4/core && /Users/signal4/.local/bin/uv sync --quiet && echo "done"' 2>&1 &
done
wait
echo "All uv syncs complete"
