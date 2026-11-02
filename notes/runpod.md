# Runpod A100 day: runbook (prepared Nov 1, run Nov 2)

Goal: the policy comparison on a modern GPU in one day, at minimum cost. Two A100 80GB pods in parallel,
one model each, both **without a CPU tier** (a miss recomputes, so a better eviction order shows in TTFT;
with LMCache a miss is a cheap reload for any model, which the 0.5B runs already showed). Each pod runs
`scripts/a100_queue.sh` unattended and stops itself when done.

| pod | setup | queue | role |
|---|---|---|---|
| `msd-14b` | `runpod_setup.sh 14b` | `a100_queue.sh nocpu 14b` | main result: a miss costs seconds, the A100 still serves many sessions |
| `msd-7b` | `runpod_setup.sh 7b` | `a100_queue.sh nocpu 7b` | second model size: does the gain grow as misses get dearer? |

Why 14B (user agreed Nov 1): the 0.5B made misses too cheap to show in TTFT; the 7B on the 2080 Ti
served too few sessions (2-3 alive) for the policy to have choices; 32B on one A100 would repeat that
(~7 GB left for KV, slow). Qwen2.5-Coder only, since the traces are tokenized with its tokenizer.
KV per token: 7B 57,344 B, 14B 196,608 B. bf16 on the A100 (`PROFILE=a100`).

Each queue: capacity probes at loads 4/8/12/16/24 (full pool, stops at the first load that falls
behind) → pools at 1.5x and 2x live KV at the highest load that kept up (`kvtier.plan`, calibrated on
the 0.5B runs) → 3 policies x 2 pools x 3 seeds. Every run is 14 min (2 warm-up, 10 measured, 2 drain).
Finished runs are skipped, so a queue can be restarted after a crash. Decisions go to
`results/replay/a100-<size>-nocpu-plan.txt`; runs to `results/replay/a100-<size>-nocpu-{probe,lru,remaining,oracle}/`.

## Cost estimate

| item | GPU-hours |
|---|---|
| setup per pod (uv sync ~8 GB, model 15-30 GB, tests) | 2 x 0.5 |
| capacity probes per pod (up to 5 x 14 min) | 2 x 1.2 |
| comparison per pod (18 x 14 min) | 2 x 4.2 |
| **total** | **~12 GPU-hours, ~6 h wall clock: ~$17-22** at $1.39-1.79/hr Secure Cloud |

Plus pod disk (100 GB each, cents for a day). The 14B's CPU-tier run (`a100_queue.sh cpu 14b`, ~5.5 h,
~$9) only if the budget allows. The real risk is an idle pod: AUTO_STOP=1 stops each pod when its queue
ends; check anyway. Prices are from third-party pages (Nov 1); confirm live in step 1.

## Steps

**0. User, on the Mac (one time).** Install the CLI, set the key, register the SSH key before any pod exists:
```bash
curl -sSL https://cli.runpod.net | bash && runpodctl update && runpodctl version
export RUNPOD_API_KEY=...            # console.runpod.io -> Settings -> API Keys
runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub && runpodctl ssh list-keys
```

**1. Check stock and price** (no cost): A100 80GB (PCIe or SXM, whichever is cheaper and in stock),
Secure Cloud, driver CUDA >= 13. Confirm syntax with `runpodctl pod create --help`.

**2. Create two pods** (user approves the price first). Pod volume disk instead of a network volume:
one day, no data-center pinning, and it survives a stop. Flags to confirm against `--help`:
```bash
for name in msd-14b msd-7b; do
  runpodctl pod create --name $name --template-id runpod-torch-v280 --gpu-id "NVIDIA A100 80GB PCIe" \
    --volume-in-gb 100 --volume-mount-path /workspace --container-disk-in-gb 30 \
    --ports "22/tcp" --ssh --min-cuda-version 13.0
done
runpodctl ssh info <pod-id>      # poll until it returns ip/port; > 6 min "not ready" = bad host, recreate
```
Then `~/.ssh/config` (the port changes after every restart; re-read `ssh info`):
```
Host msd-14b
    HostName <ip>
    Port <port>
    User root
    IdentityFile <ssh_key.path from ssh info>
    ServerAliveInterval 30
```
(and the same for `msd-7b`).

**3. Bootstrap, code, traces, setup** (per pod `P` with size `S`: msd-14b/14b, msd-7b/7b):
```bash
ssh P 'apt-get update -qq && apt-get install -y -qq rsync tmux'
GPU_HOST=P GPU_DIR=/workspace/msd_project scripts/remote.sh true          # sync the code
ssh gpubox 'tar -C msd_project -czf - traces/raw traces/sessions' | ssh P 'tar -C /workspace/msd_project -xzf -'
GPU_HOST=P GPU_DIR=/workspace/msd_project scripts/remote.sh scripts/runpod_setup.sh S   # ends with "== ready"
ssh P 'echo pod=$RUNPOD_POD_ID; command -v runpodctl'   # AUTO_STOP needs both (see below)
```

**4. Start the queues:**
```bash
GPU_HOST=msd-14b GPU_DIR=/workspace/msd_project scripts/remote.sh \
  "tmux new -d -s exp0 'AUTO_STOP=1 RUNPOD_POD_ID=<14b pod id> scripts/a100_queue.sh nocpu 14b 2>&1 | tee logs/queue.log'"
# same on msd-7b with `nocpu 7b`
```
RUNPOD_POD_ID is passed explicitly because an SSH shell may not see the container's environment. If
`runpodctl` on the pod has no key (step 3 check fails), drop AUTO_STOP and stop pods from the Mac.

**5. Watch** (cheap reads; the first probe tells whether the setup is sane):
```bash
GPU_HOST=msd-14b GPU_DIR=/workspace/msd_project scripts/remote.sh "tail -3 logs/queue.log; cat results/replay/a100-14b-nocpu-plan.txt"
```
Sanity after the first probe: `returning_slo_share_2s` high, `pause_share` well above the 2080 Ti's
30%, and the server log shows the FlashAttention backend (not Triton). After the first `remaining` run:
`kvtier: evicted ... orphan blocks, ... by policy` lines in `logs/vllm0.log`.

**6. Collect, then delete** (results go to gpubox, the home of all results):
```bash
mkdir -p results/a100 && rsync -az msd-14b:/workspace/msd_project/results/replay/ results/a100/ \
  && rsync -az msd-7b:/workspace/msd_project/results/replay/ results/a100/     # results/ is gitignored
rsync -az results/a100/ gpubox:msd_project/results/replay/
runpodctl pod remove <id>        # both pods, only after the copy is confirmed on gpubox
```
Then compare on gpubox (`results/replay/a100-{7b,14b}-nocpu-{lru,remaining,oracle}`).

## What differs from gpubox

- 14B/7B in bf16 on stock vLLM attention: `PROFILE=a100` sets `KVTIER_SDPA_PREFILL=0`, `DTYPE=bfloat16`.
- `scripts/emu_run.sh` waits for GPU memory to drop, not for process IDs (containers hide them).
- Only `/workspace` survives a stop: `remote.sh` keeps uv's Python and caches and `HF_HOME` there.
- Rehearsed end to end on gpubox with the 0.5B (`NAME=dryrun`, short runs): see `logs/dryrun.log`.
