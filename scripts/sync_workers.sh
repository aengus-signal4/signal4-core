#!/bin/bash
# Sync code and dependencies to all workers
# Usage: ./sync_workers.sh [targets]
#   No args     - sync to all workers
#   3           - sync to worker3 only
#   3,4         - sync to worker3 and worker4
#   10.0.0.34   - sync to specific IP

declare -A WORKER_IPS=(
  [0]="10.0.0.34"
  [1]="10.0.0.9"
  [2]="10.0.0.203"
  [3]="10.0.0.51"
  [4]="10.0.0.209"
)
ALL_WORKERS="${WORKER_IPS[@]}"
SOURCE_DIR="/Users/signal4/signal4/core/"

if [ -n "$1" ]; then
  WORKERS=""
  # Split by comma and resolve each target
  IFS=',' read -ra TARGETS <<< "$1"
  for target in "${TARGETS[@]}"; do
    if [[ "$target" =~ ^[0-9]+$ ]] && [ -n "${WORKER_IPS[$target]}" ]; then
      WORKERS="$WORKERS ${WORKER_IPS[$target]}"
      echo "Targeting worker$target (${WORKER_IPS[$target]})"
    elif [[ "$target" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      WORKERS="$WORKERS $target"
      echo "Targeting IP: $target"
    else
      echo "Unknown target: $target (use 0-4 or IP address)"
      exit 1
    fi
  done
else
  WORKERS="$ALL_WORKERS"
fi

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
