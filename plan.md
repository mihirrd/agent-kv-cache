# 3-Week Implementation Plan (Oct 19 – Nov 8, 2026)

Compresses SPEC.md's 7-week plan into 3 weeks, through results and figures. The spec's own
scope cuts are applied up front; components C1–C8 and gates refer to SPEC.md.

## Changes from the spec

- **gpubox has no Docker, Python 3.8, and `uv` not confirmed on PATH.** Install uv + a newer Python
  on day 1. Recording SWE-bench traces needs Docker (sudo); otherwise use the spec's fallback of
  running mini-SWE-agent in plain venvs.
- **GPUs are 2080 Ti (Turing, 11 GB, fp16 only).** Newer vLLM releases don't support the FlashAttention
  backend on Turing, so day 1 must confirm a V1 attention backend runs on sm_75 together with LMCache.
  This is the new top risk.
- **Model: Qwen2.5-Coder-7B-Instruct-AWQ, TP=2 across both GPUs (NVLink, 2 links ≈ 52 GB/s each
  way).** Decided 2026-10-20 after Gate A measurements: TP=2 gives a 206K-token GPU KV pool (vs
  36K on one GPU) and roughly halves prefill (8K cold: 7.6 s vs 11.9 s; 16K: ~29 s vs ~50 s).
  KV is ~56 KB/token (28 layers × 4 KV heads × 128 dim × K and V × 2 bytes). The memory cap is
  set with `--gpu-memory-utilization`. GPU 0 also drives the desktop, so watch for noise.
- **Contexts capped at ~16K tokens.** TRITON_ATTN on Turing slows down roughly quadratically with
  prompt length (28K cold prefill: 82–99 s even with TP=2), so 30K contexts don't fit the run budget.
- **Scope cuts applied now:** no disk tier, noisy oracle at ±50% only, three load levels instead
  of five, Continuum fork replaced by predicted-TTL pinning written as a `KVPolicy`. Prefetch stays
  only if its gate passes.
- **Simulator runs on gpubox's CPU**, not the Mac (remote-only rule). Recording on gpubox's x86
  CPUs removes the x86-emulation blocker.

## Week 1 (Oct 19–25): stack, recording, offline components

| Day | Work | Gate or output |
|---|---|---|
| Mon | `uv` + Python 3.11 env, pin vLLM to LMCache's supported version, `third_party/vllm` editable install with `VLLM_USE_PRECOMPILED=1`. Serve 0.5B, then 7B-AWQ (TP=2). | **Gate A:** vLLM V1 + LMCache serve on sm_75. If it fails, try another attention backend or an older pinned version. |
| Tue | Check that the session ID reaches the engine (`user` field vs header vs `extra_args`). Test the warm-up request (last prompt, `max_tokens=1`), then confirm the real request gets a prefix-cache hit on Prometheus. | **Gate B:** prefetch stays in scope or is cut. |
| Tue–Wed | Install Docker or set up the venv fallback. Write the C5 recorder wrapping mini-SWE-agent with LiteLLM. Check API cost after 10 tasks. Start recording in tmux `exp`, 60–80 tasks, steps capped so contexts stay under ~16K tokens. | Recording runs in the background for the rest of the week. |
| Wed–Thu | Tests first, then `KVPolicy`, `Session`, the C2 predictor (back-off keys, ring buffer of 50, p10/p50/p90, confidence rule) and the C3 cost model. | `pytest -m "not gpu"` passes on gpubox. |
| Thu–Sat | C7 simulator: event heap, GPU/CPU block pools. Policies: LRU, always-offload, oracle, TTL-pin. Placeholder calibration constants for now. | Simulator runs a policy on a sample of traces in seconds. |
| Sat–Sun | Recording finishes. Hold out 20% of sessions. Run the simulator sweep at the tight cap. | **Gate C (headroom):** oracle beats always-offload by ≥5% goodput. If not, try a tighter cap or longer contexts; if still not, reframe as a characterization study. |

**End of week 1:** versions pinned and serving, Gates A–C decided, traces recorded, predictor and
simulator tested.

## Week 2 (Oct 26 – Nov 1): harness, calibration, vLLM integration

| Day | Work | Output |
|---|---|---|
| Mon | C6 replay harness: asyncio + httpx, recorded messages, `min_tokens`/`max_tokens` forced, recorded tool sleeps, Poisson arrivals. C8 metrics logging and a Prometheus scraper every 5 s. | A GPU-marked test checks replayed prompts match the trace token for token. |
| Tue | Calibration at TP=2: prefill tokens/s by prompt length, CPU↔GPU GB/s, CPU-reload TTFT after warm-up at several sizes (one TP=2 sample gave 0.95 s at 8K vs 0.27 s on one GPU; find out why), decode time per token. Choose the TTFT SLO and three load levels (lowest meets the SLO under vLLM default, highest violates it). Pilot variance check. | Calibration JSON that the simulator also reads. Flag if pilot runs differ by >10%. |
| Tue–Wed | Measure vLLM default and always-offload baselines (one pass each) to validate the pipeline. Rerun the simulator with real calibration numbers. | First real goodput numbers. |
| Wed–Fri | C1 inside vLLM: `SessionTable`, parse the program name from the last fenced bash block, mark PAUSED in the scheduler's finish path, match returning requests by session ID + first-block hash. Policy-driven eviction in `block_pool.py` / `kv_cache_manager.py`, run with the **oracle**. Test with 0.5B first. | Oracle policy running in vLLM. |
| Sat | Compare simulator vs real runs for oracle and always-offload at matching load and cap. | **Gate D:** within 15% → use the simulator for tuning; otherwise tune on real runs only and report the gap. |
| Sun | Buffer and fixes. | |

## Week 3 (Nov 2–8): full system, experiments, figures

| Day | Work | Output |
|---|---|---|
| Mon | Connect the C2 predictor online: `on_return` feeds measured pauses back; low-confidence sessions fall back to LRU. Write the TTL-pin baseline. C4 prefetch (if Gate B passed): fire at pause start + p10 − reload time, cancel if the agent already returned, protect the warm-up's blocks from eviction. | All 5 configs + noisy oracle run end to end on 0.5B. |
| Tue | Predictor calibration on the held-out 20%: share of true pauses inside p10–p90, target ≥75%. Tune thresholds in the simulator. Sweep driver that restarts the server between runs and writes to `results/<config>/<load>/<cap>/<run>/`. | Predictor calibration table. |
| Tue night → Thu | Full sweep: 5 configs × 3 loads × 2 caps × 3 runs, plus noisy oracle (±50%) × 3 ≈ 93 runs × 12 min ≈ 19 GPU-hours. TP=2 uses both GPUs, so runs are sequential (~2 nights). | All runs done, logs archived. |
| Fri | Figure scripts on gpubox: goodput vs load per cap, returning-turn TTFT p50/p99, re-prefilled GPU-seconds, goodput vs injected error, predictor calibration, simulator vs real. | `results/figures/` |
| Sat–Sun | Check success criteria, rerun noisy cells, start the write-up. | |

## Slack and cuts

- **Most likely slip:** week 2's vLLM integration (Wed–Fri). If it slips, cut prefetch first, then
  the noisy oracle. The core claim (oracle and predicted placement vs vLLM default and
  always-offload) needs only C1–C3.
- **Parallel work:** recording overlaps predictor/simulator work in week 1; the simulator runs on
  CPU while GPU runs are going.
- **Commit after each passed gate**, per the CLAUDE.md workflow.
