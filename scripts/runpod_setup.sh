#!/usr/bin/env bash
# Get a Runpod pod ready for experiments; run after every pod start (a stop wipes everything outside
# /workspace). Safe to re-run. See notes/runpod.md.
#   GPU_HOST=<pod> GPU_DIR=/workspace/msd_project scripts/remote.sh scripts/runpod_setup.sh 14b   (or 7b)
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/.uv-cache}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-/workspace/.uv-python}
export HF_HOME=${HF_HOME:-/workspace/hf}
case ${1:?usage: scripts/runpod_setup.sh 7b|14b} in
  7b) MODEL=Qwen/Qwen2.5-Coder-7B-Instruct ;;
  14b) MODEL=Qwen/Qwen2.5-Coder-14B-Instruct ;;
  *) echo "size must be 7b or 14b" >&2; exit 1 ;;
esac
mkdir -p logs

echo "== system packages"
if ! command -v tmux >/dev/null || ! command -v rsync >/dev/null; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux rsync curl >/dev/null
fi

echo "== GPU and driver (LMCache 0.5.6 needs a CUDA 13 driver)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
cuda=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$')
echo "driver CUDA $cuda"
[[ ${cuda%%.*} -ge 13 ]] || { echo "driver CUDA $cuda < 13: recreate the pod with --min-cuda-version 13.0" >&2; exit 1; }

echo "== uv and the Python environment"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
uv python install 3.11
uv sync 2>&1 | tee logs/uv_sync.log | tail -3

echo "== model ($MODEL)"
uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"

echo "== traces"
n=$(find traces/raw -name '*.jsonl' 2>/dev/null | wc -l)
echo "$n raw trace files"
[[ $n -gt 0 ]] || { echo "traces/raw is empty: copy it from the Mac (notes/runpod.md, step 4)" >&2; exit 1; }

echo "== fast tests"
uv run pytest -q -m "not gpu" 2>&1 | tail -2
echo "== ready"
