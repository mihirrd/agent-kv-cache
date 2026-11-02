#!/usr/bin/env bash
# Sync the working tree to gpubox, then run a command there from the project root.
# Usage: scripts/remote.sh <command...>
# A Runpod pod instead: GPU_HOST=runpod GPU_DIR=/workspace/msd_project scripts/remote.sh <command...>
set -euo pipefail
HOST=${GPU_HOST:-gpubox}
DIR=${GPU_DIR:-msd_project}   # relative to the remote home dir

cd "$(dirname "$0")/.."

rsync -az --delete \
  --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
  --exclude results/ --exclude logs/ --exclude traces/raw/ \
  --exclude '*.docx' \
  --filter 'P third_party/vllm/**' --filter 'P traces/' \
  ./ "$HOST:$DIR/"

# On a Runpod pod (notes/runpod.md) only /workspace survives a stop, so uv's Python, its cache and the
# model cache live there.
ssh "$HOST" "export PATH=\$HOME/.local/bin:\$PATH; [ -d /workspace ] && export UV_CACHE_DIR=/workspace/.uv-cache \
UV_PYTHON_INSTALL_DIR=/workspace/.uv-python HF_HOME=/workspace/hf; cd $DIR && $*"
