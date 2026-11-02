#!/usr/bin/env bash
# Mac side: wait for a pod's queue to end, copy its results to the Mac and on to gpubox, then stop the
# pod. Replaces AUTO_STOP when the pod has no Runpod API key (Nov 2: Runpod pods have none in SSH
# shells). Copies every check, so a crash loses at most one check's results. See notes/runpod.md.
# Usage: scripts/pod_watch.sh <ssh alias> <pod id> [check every seconds, default 300]
set -uo pipefail
cd "$(dirname "$0")/.."
HOST=$1 POD=$2 EVERY=${3:-300}
DIR=/workspace/msd_project
mkdir -p results/a100

copy() {
  rsync -az "$HOST:$DIR/results/replay/" results/a100/ && rsync -az "$HOST:$DIR/logs/" "results/a100/logs-$HOST/"
}

while :; do
  state=$(ssh -o ConnectTimeout=20 "$HOST" "tmux has-session -t =exp0 2>/dev/null && echo running || echo ended" 2>/dev/null)
  copy || echo "[watch] $(date) copy failed"
  if [[ "$state" == ended ]]; then
    echo "[watch] $(date) queue ended on $HOST"
    tail -3 "results/a100/logs-$HOST/queue.log"
    copy && rsync -az --exclude 'logs-*' results/a100/ gpubox:msd_project/results/replay/ \
      && echo "[watch] results on gpubox"
    runpodctl pod stop "$POD" && echo "[watch] $(date) pod $POD stopped"
    exit 0
  fi
  [[ -z "$state" ]] && echo "[watch] $(date) $HOST unreachable"
  sleep "$EVERY"
done
