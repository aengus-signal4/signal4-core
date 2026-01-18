#!/bin/bash
# Fix PATH on all workers - remove conda activation, ensure Homebrew ffmpeg is used
# This script:
# 1. Removes conda initialization from ~/.zshrc
# 2. Ensures Homebrew ffmpeg is installed
# 3. Verifies the correct ffmpeg is in PATH

declare -A WORKER_IPS=(
  [0]="10.0.0.34"
  [1]="10.0.0.9"
  [3]="10.0.0.203"
  [4]="10.0.0.51"
  [5]="10.0.0.209"
  [6]="10.0.0.4"
)

# Allow targeting specific workers
if [ -n "$1" ]; then
  WORKERS=""
  IFS=',' read -ra TARGETS <<< "$1"
  for target in "${TARGETS[@]}"; do
    if [[ "$target" =~ ^[0-9]+$ ]] && [ -n "${WORKER_IPS[$target]}" ]; then
      WORKERS="$WORKERS ${WORKER_IPS[$target]}"
      echo "Targeting worker$target (${WORKER_IPS[$target]})"
    elif [[ "$target" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      WORKERS="$WORKERS $target"
      echo "Targeting IP: $target"
    else
      echo "Unknown target: $target"
      exit 1
    fi
  done
else
  WORKERS="${WORKER_IPS[@]}"
fi

for ip in $WORKERS; do
  echo ""
  echo "=== Fixing PATH on $ip ==="

  ssh -o StrictHostKeyChecking=no signal4@$ip bash << 'EOF'
    # Remove conda initialization from .zshrc
    if grep -q "conda activate\|conda.sh" ~/.zshrc 2>/dev/null; then
      echo "Removing conda activation from ~/.zshrc..."
      # Create backup
      cp ~/.zshrc ~/.zshrc.backup
      # Remove conda lines
      sed -i '' '/source.*conda.sh/d' ~/.zshrc
      sed -i '' '/conda activate/d' ~/.zshrc
      echo "Conda lines removed from ~/.zshrc"
    else
      echo "No conda activation found in ~/.zshrc"
    fi

    # Ensure ~/.local/bin is in PATH (for uv)
    if ! grep -q 'HOME/.local/bin' ~/.zshrc 2>/dev/null; then
      echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
      echo "Added ~/.local/bin to PATH"
    fi

    # Check Homebrew ffmpeg
    if [ -f /opt/homebrew/bin/ffmpeg ]; then
      echo "Homebrew ffmpeg found: $(/opt/homebrew/bin/ffmpeg -version 2>&1 | head -1)"
    else
      echo "WARNING: Homebrew ffmpeg not found. Install with: brew install ffmpeg"
    fi

    # Verify PATH in new shell
    echo ""
    echo "Verifying PATH (from new shell):"
    /bin/zsh -c 'source ~/.zshrc 2>/dev/null; which ffmpeg; which uv; which python'
EOF
done

echo ""
echo "=== PATH fix complete ==="
echo "Workers may need their processors restarted for changes to take effect."
