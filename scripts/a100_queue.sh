#!/usr/bin/env bash
# The whole rented-GPU experiment for one CPU-tier mode, with no human step. Run inside tmux `exp0`:
#   scripts/a100_queue.sh nocpu 14b    # no CPU tier, a miss recomputes: the main result (one pod)
#   scripts/a100_queue.sh nocpu 7b     # the same with the 7B (the other pod)
#   scripts/a100_queue.sh cpu 14b      # LMCache CPU tier on, a miss reloads (only if budget allows)
# 1. capacity: full KV pool at rising loads, stopping at the first that falls behind;
# 2. pools: 1.5x and 2x the live KV of the full-pool run at the highest load that kept up (kvtier.plan);
# 3. comparison: LRU / remaining-time / oracle x 2 pools x seeds.
# Finished runs are skipped, so it resumes after a crash. AUTO_STOP=1 stops the pod when the queue
# exits, finished or failed (results stay on /workspace). See notes/runpod.md.
# Overrides: PROFILE (a100), MODEL, NAME, LOADS, SEEDS, FRACTIONS, REPLAY_ARGS (e.g. "--warmup 10 --duration 60").
# POLICIES (default "lru remaining oracle"; also continuum, remaining+continuum). LOAD and POOLS
# together skip steps 1-2 and reuse an earlier plan, e.g. LOAD=8 POOLS="82000 109000".
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:?usage: scripts/a100_queue.sh cpu|nocpu 7b|14b}
SIZE=${2:?usage: scripts/a100_queue.sh cpu|nocpu 7b|14b}
case $MODE in
  cpu) export NO_LMCACHE= ;;
  nocpu) export NO_LMCACHE=1 ;;
  *) echo "mode must be cpu or nocpu" >&2; exit 1 ;;
esac
case $SIZE in
  7b) export MODEL=${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct} ;;
  14b) export MODEL=${MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct} ;;
  *) echo "size must be 7b or 14b" >&2; exit 1 ;;
esac
export PROFILE=${PROFILE:-a100}
read -ra LOADS <<< "${LOADS:-4 8 12 16 24}"
read -ra SEEDS <<< "${SEEDS:-0 1 2}"
read -ra FRACTIONS <<< "${FRACTIONS:-1.5 2}"
read -ra EXTRA <<< "${REPLAY_ARGS:-}"
read -ra POLICIES <<< "${POLICIES:-lru remaining oracle}"
NAME=${NAME:-a100-$SIZE-$MODE}
PROBE=$NAME-probe
PLAN=results/replay/$NAME-plan.txt
mkdir -p results/replay

stop_pod() {
  [[ "${AUTO_STOP:-}" == 1 && -n "${RUNPOD_POD_ID:-}" ]] || return 0
  echo "[queue] $(date -Is) stopping pod $RUNPOD_POD_ID"
  runpodctl pod stop "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
}
trap stop_pod EXIT

done_run() { [[ -e results/replay/$1/load$2/$3/seed$4/summary.json ]]; }

if [[ -n "${LOAD:-}" && -n "${POOLS:-}" ]]; then
  best=$LOAD
  read -ra POOLS <<< "$POOLS"
else
  echo "[queue] $(date -Is) $MODEL, $MODE: capacity probes at loads ${LOADS[*]}" | tee -a "$PLAN"
  for load in "${LOADS[@]}"; do
    done_run "$PROBE" "$load" full 0 || KVTIER_POLICY= scripts/emu_run.sh 0 full "$load" "$PROBE" --seed 0 "${EXTRA[@]}"
    best=$(uv run python -m kvtier.plan load "results/replay/$PROBE" "${LOADS[@]}")
    [[ "$best" == "$load" ]] || break  # this load fell behind; the probes above it would too
  done
  best=$(uv run python -m kvtier.plan load "results/replay/$PROBE" "${LOADS[@]}")
  [[ -n "$best" ]] || { echo "[queue] no probed load keeps up; probe lower loads" | tee -a "$PLAN" >&2; exit 1; }

  read -ra POOLS <<< "$(uv run python -m kvtier.plan pools "results/replay/$PROBE/load$best/full/seed0" "${FRACTIONS[@]}" 2>>"$PLAN")"
  for i in "${!POOLS[@]}"; do
    # vLLM needs one longest replayed request (~17K tokens) to fit in the pool.
    (( POOLS[i] >= 20000 )) || POOLS[i]=20000
  done
fi
echo "[queue] $(date -Is) load $best, pools ${POOLS[*]}, seeds ${SEEDS[*]}, policies ${POLICIES[*]}" | tee -a "$PLAN"

# policy name -> server env: eviction order (KVTIER_POLICY) and Continuum's scheduling (KVTIER_SCHED)
policy_env() {
  case $1 in
    continuum) echo "KVTIER_POLICY= KVTIER_SCHED=continuum" ;;
    *+continuum) echo "KVTIER_POLICY=${1%+continuum} KVTIER_SCHED=continuum" ;;
    *) echo "KVTIER_POLICY=$1 KVTIER_SCHED=" ;;
  esac
}

for seed in "${SEEDS[@]}"; do
  for pool in "${POOLS[@]}"; do
    for policy in "${POLICIES[@]}"; do
      # shellcheck disable=SC2046  # policy_env's words are VAR=value pairs
      done_run "$NAME-$policy" "$best" "$pool" "$seed" ||
        env $(policy_env "$policy") scripts/emu_run.sh 0 "$pool" "$best" "$NAME-$policy" --seed "$seed" "${EXTRA[@]}"
    done
  done
done
echo "[queue] $(date -Is) all done" | tee -a "$PLAN"
