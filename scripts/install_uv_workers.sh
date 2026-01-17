#!/bin/bash
# Install uv on all workers that don't have it

WORKERS="10.0.0.34 10.0.0.203 10.0.0.51 10.0.0.209"

for ip in $WORKERS; do
  echo "Installing uv on $ip..."
  ssh -o StrictHostKeyChecking=no signal4@$ip 'curl -LsSf https://astral.sh/uv/install.sh | sh' 2>&1 &
done
wait
echo "All uv installs complete"
