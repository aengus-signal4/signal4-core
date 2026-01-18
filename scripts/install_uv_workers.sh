#!/bin/bash
# Install uv and required dependencies on all workers
# This script installs:
# - uv (Python package manager)
# - ffmpeg (required for audio processing)

declare -A WORKER_IPS=(
  [0]="10.0.0.34"
  [1]="10.0.0.9"
  [3]="10.0.0.203"
  [4]="10.0.0.51"
  [5]="10.0.0.209"
)

# Allow targeting specific workers
if [ -n "$1" ]; then
  WORKERS=""
  IFS=',' read -ra TARGETS <<< "$1"
  for target in "${TARGETS[@]}"; do
    if [[ "$target" =~ ^[0-9]+$ ]] && [ -n "${WORKER_IPS[$target]}" ]; then
      WORKERS="$WORKERS ${WORKER_IPS[$target]}"
      echo "Targeting worker$target (${WORKER_IPS[$target]})"
    fi
  done
else
  WORKERS="${WORKER_IPS[@]}"
fi

for ip in $WORKERS; do
  echo ""
  echo "=== Setting up $ip ==="

  ssh -o StrictHostKeyChecking=no signal4@$ip bash << 'EOF'
    # Install uv if not present
    if ! command -v uv &> /dev/null; then
      echo "Installing uv..."
      curl -LsSf https://astral.sh/uv/install.sh | sh
    else
      echo "uv already installed: $(uv --version)"
    fi

    # Install Homebrew ffmpeg if not present
    if ! /opt/homebrew/bin/ffmpeg -version &> /dev/null; then
      echo "Installing ffmpeg via Homebrew..."
      brew install ffmpeg
    else
      echo "ffmpeg already installed: $(/opt/homebrew/bin/ffmpeg -version 2>&1 | head -1)"
    fi
EOF
done &

wait
echo ""
echo "=== All installations complete ==="
