#!/usr/bin/env bash
# Start the vLLM OpenAI server with LMCache CPU offload. Run on gpubox inside tmux `vllm`.
# Overrides: MODEL, GPU, PORT, GPU_MEM_UTIL, MAX_MODEL_LEN, CPU_CACHE_GB, ATTN_BACKEND, NO_LMCACHE=1.
# Extra CLI args are passed through to `vllm serve`.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen2.5-Coder-0.5B-Instruct}
GPU=${GPU:-1}
PORT=${PORT:-8000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

export CUDA_VISIBLE_DEVICES=$GPU
export LMCACHE_CHUNK_SIZE=${LMCACHE_CHUNK_SIZE:-256}
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=${CPU_CACHE_GB:-20}

args=(
  "$MODEL"
  --port "$PORT"
  --dtype float16
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --enable-prefix-caching
)
[[ -n "${ATTN_BACKEND:-}" ]] && args+=(--attention-backend "$ATTN_BACKEND")
[[ -z "${NO_LMCACHE:-}" ]] && args+=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')

echo "[run_vllm] $(date -Is) GPU=$GPU model=$MODEL args: ${args[*]} $*"
exec uv run --no-sync vllm serve "${args[@]}" "$@"
