#!/usr/bin/env bash
# Sync the working tree to gpubox, then run a command there from the project root.
# Usage: scripts/remote.sh <command...>
set -euo pipefail
HOST=${GPU_HOST:-gpubox}
DIR=${GPU_DIR:-msd_project}   # relative to the remote home dir

cd "$(dirname "$0")/.."

rsync -az --delete \
  --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
  --exclude results/ --exclude logs/ --exclude traces/raw/ \
  --exclude '*.docx' \
  --filter 'P third_party/vllm/**' \
  ./ "$HOST:$DIR/"

ssh "$HOST" "export PATH=\$HOME/.local/bin:\$PATH; cd $DIR && $*"
