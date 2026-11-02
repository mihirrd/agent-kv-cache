#!/usr/bin/env bash
# One emulator experiment cell on gpubox: restart the GPU's 0.5B server with a fixed KV pool, wait for
# it, then replay. Run inside tmux (exp0 / exp1). See notes/emulation.md.
# Usage: [KVTIER_POLICY=remaining|oracle] [KVTIER_SCHED=continuum] [NO_LMCACHE=1] [PROFILE=7b1] [MODEL=...] scripts/emu_run.sh <gpu> <pool_tokens|full> <load_per_min> <config> [extra replay args...]
# KVTIER_POLICY unset is stock vLLM (LRU); see third_party/vllm/vllm/v1/core/kvtier_sessions.py.
# KVTIER_SCHED=continuum adds the Continuum baseline's pins and queue order (kvtier_continuum.py).
# Server env (e.g. KVTIER_SDPA_PREFILL=0) passes through. Output: results/replay/<config>/load<L>/<pool>/seed<S>/
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=$1 POOL=$2 LOAD=$3 CONFIG=$4
shift 4
PORT=$((8000 + GPU))
BLOCKS=""
MAX_LEN=32768
if [[ "$POOL" != full ]]; then
  BLOCKS=$(( (POOL + 15) / 16 ))
  # vLLM refuses to start unless one max-length request fits in the pool. Replayed requests stay under
  # ~17K tokens (15.5K prompt cap + reply), so pools must be at least ~20K.
  # One block is reserved by vLLM, hence BLOCKS - 1.
  MAX_LEN=$(( (BLOCKS - 1) * 16 < 32768 ? (BLOCKS - 1) * 16 : 32768 ))
fi

mkdir -p logs
tmux kill-session -t "vllm$GPU" 2>/dev/null || true
# Wait until the old server has released the GPU. Memory, not process IDs: inside a container
# (Runpod) nvidia-smi cannot see the processes. GPU 0 on gpubox holds ~200 MiB for the desktop.
while (( $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU") > 1024 )); do sleep 2; done
tmux new -d -s "vllm$GPU" \
  "KVTIER_POLICY=${KVTIER_POLICY:-} KVTIER_SCHED=${KVTIER_SCHED:-} NO_LMCACHE=${NO_LMCACHE:-} PROFILE=${PROFILE:-emu} MODEL=${MODEL:-} GPU=$GPU NUM_GPU_BLOCKS=$BLOCKS MAX_MODEL_LEN=$MAX_LEN ./scripts/run_vllm.sh 2>&1 | tee logs/vllm$GPU.log"
until curl -sf "localhost:$PORT/health" >/dev/null; do
  tmux has-session -t "vllm$GPU" 2>/dev/null || { echo "server on GPU $GPU died; see logs/vllm$GPU.log" >&2; exit 1; }
  sleep 3
done

exec uv run python -m kvtier.replay run --base-url "http://localhost:$PORT" --load "$LOAD" --config "$CONFIG" \
  --cap "$POOL" "$@"
