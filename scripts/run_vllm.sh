#!/usr/bin/env bash
# Start the vLLM OpenAI server with LMCache CPU offload. Run on gpubox inside tmux `vllm`.
# Overrides: PROFILE, MODEL, GPU (e.g. 1 or 0,1), TP, PORT, GPU_MEM_UTIL, MAX_MODEL_LEN, CPU_CACHE_GB,
# NUM_GPU_BLOCKS (fixed KV pool, 16 tokens per block), ATTN_BACKEND, NO_LMCACHE=1.
# Extra CLI args are passed through to `vllm serve`.
#
# PROFILE=emu (the default experiment setup, see notes/emulation.md): 0.5B fp16 on one GPU, so each
#   GPU runs its own server (GPU 0 -> port 8000, GPU 1 -> port 8001). Set NUM_GPU_BLOCKS to fix the pool.
# PROFILE=7b: 7B-AWQ, TP=2 across both GPUs, port 8000.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${PROFILE:-}" in
  emu)
    MODEL=${MODEL:-Qwen/Qwen2.5-Coder-0.5B-Instruct}
    GPU=${GPU:-1}
    TP=${TP:-1}
    PORT=${PORT:-$((8000 + ${GPU%%,*}))}
    ;;
  7b)
    MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct-AWQ}
    ;;
  "") ;;
  *) echo "unknown PROFILE=$PROFILE (emu or 7b)" >&2; exit 1 ;;
esac

MODEL=${MODEL:-Qwen/Qwen2.5-Coder-0.5B-Instruct}
GPU=${GPU:-0,1}
TP=${TP:-2}
PORT=${PORT:-8000}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}

export CUDA_VISIBLE_DEVICES=$GPU
# LMCache keys chunks with Python's hash(); with TP>1 the scheduler and workers are separate
# processes, so an unset seed gives each its own hashes and every LMCache lookup misses.
export PYTHONHASHSEED=0
# Mixed decode+prefill steps: decodes via the Triton 3D path, prefills via PyTorch SDPA (our vLLM fork,
# triton_attn.py). Prefill attention is ~5-10x faster on Turing. KVTIER_SDPA_PREFILL=0 restores stock vLLM.
export KVTIER_SDPA_PREFILL=${KVTIER_SDPA_PREFILL:-1}
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
  --tensor-parallel-size "$TP"
  --enable-prompt-tokens-details
)
[[ -n "${NUM_GPU_BLOCKS:-}" ]] && args+=(--num-gpu-blocks-override "$NUM_GPU_BLOCKS")
[[ -n "${ATTN_BACKEND:-}" ]] && args+=(--attention-backend "$ATTN_BACKEND")
[[ -z "${NO_LMCACHE:-}" ]] && args+=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')

echo "[run_vllm] $(date -Is) GPU=$GPU TP=$TP model=$MODEL args: ${args[*]} $*"
exec uv run --no-sync vllm serve "${args[@]}" "$@"
