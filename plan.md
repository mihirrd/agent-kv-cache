# 3-Week Implementation Plan (Oct 19 – Nov 8, 2026)

Compresses SPEC.md's 7-week plan into 3 weeks, through results and figures. Components C1–C8 and
gates refer to SPEC.md. **Revised Oct 22** after Gate C: the project is now a caching (eviction)
policy, compared against Continuum. Measurements and reasons: `notes/week1-findings.md`.

## The research question (revised Oct 22)

When GPU memory runs out, which paused agent's KV should be evicted? Our policy, **remaining-time
eviction**: keep KV while there is room; under pressure, evict the paused session expected to stay
away longest, re-estimating as the pause goes on (past pauses of the same tool that lasted longer
than this one so far). Returning sessions reload from the LMCache CPU copy.

Simulator evidence (placeholder calibration): +7% to +11% goodput over always-offload where memory
runs out, never negative; a perfect guesser (Belady) gets up to +34%. Continuum's gain comes from its
queue order, not its TTL pinning; ours beats pinning and combines with its queue order.

## Changes from the spec

- **Hardware and serving (decided week 1):** 2× RTX 2080 Ti, Qwen2.5-Coder-7B-Instruct-AWQ, fp16,
  TP=2 (206K-token KV pool), `TRITON_ATTN`, vLLM 0.31.0 + LMCache 0.5.6, contexts capped at ~16K tokens.
- **Dropped:** the C3 KEEP/OFFLOAD/DISCARD rule and proactive offload. They lose to always-offload
  even with perfect knowledge, because offloading early frees memory nobody needs yet.
- **Demoted to stretch goal:** C4 prefetch. Warm-ups lost in every simulator test; revisit only if
  calibration shows reloads are cheap and don't stall the batch.
- **C2 changes role:** it now estimates *time left* in a pause given time already elapsed
  (`PausePredictor.remaining`), which drives eviction order.
- **Continuum:** reimplemented as a policy (TTL pinning + program-level FCFS queue order), not run
  from its fork (it needs vLLM 0.10.2). It is the main comparison.
- **New risk: our GPUs are compute-bound.** On the 2080 Tis, compute runs out before memory, so the
  eviction gain may not show in real runs. Gate E (week 2) decides how to get a memory-bound setup.
- **Scope kept from the spec's cuts:** no disk tier; three load levels; the simulator runs on gpubox's CPU.

## Week 1 (Oct 19–25): stack, recording, simulator — done ahead of plan

| Day | Work | Result |
|---|---|---|
| Mon | Stack: driver 580, Python 3.11, vLLM 0.31.0 + LMCache 0.5.6, 7B-AWQ at TP=2 | **Gate A: PASS** |
| Tue | Session ID via `vllm_xargs`; warm-up request gives a GPU prefix hit | **Gate B: PASS** |
| Tue–Wed | C5 recorder (mini-SWE-agent + Haiku 5.5, testing workflow); 50 Verified + 20 Multilingual traces | 70 traces, $0.38 |
| Wed–Thu | `KVPolicy`, C2 predictor, C3 cost model, replay-session preprocessing | tested |
| Thu | C7 simulator; Gate C; long-call offload; faster-GPU regime; remaining-time eviction; Continuum head-to-head | **Gate C: fails on today's GPUs; eviction order pays when memory is the bottleneck** |
| Thu | Revise this plan | done |
| Fri | Start C6 replay harness (pulled from week 2) | |
| Sat | Start calibration with the harness (pulled from week 2) | |
| Sun | Improve the time-left guess in the simulator, e.g. key on the command's target (test file, directory), not only the program name | closer to Belady |

## Week 2 (Oct 26 – Nov 1): measure, integrate, validate

| Day | Work | Output / gate |
|---|---|---|
| Mon | Finish C6 harness: asyncio + httpx, recorded prompts, `min_tokens`/`max_tokens` forced, recorded tool sleeps, Poisson arrivals, session ID in `vllm_xargs`. C8 metrics and Prometheus scraping every 5 s. | GPU test: replayed prompts match the trace token for token. |
| Tue | Calibration at TP=2: incremental prefill (new tokens × context grid), decode step time by batch size, CPU-reload TTFT with warm-up and repeats at several sizes, **and whether a reload stalls other requests in the batch**. Pilot variance check. | Calibration JSON read by the simulator. Re-run the key simulator comparisons with it. |
| Tue–Wed | Baselines on real hardware: vLLM default and always-offload, one pass each. | First real goodput numbers. |
| Wed–Fri | **C1 in vLLM:** a session table (session ID → blocks, pause start, program); at a turn's finish, record the pause start and tool; **eviction order in `block_pool.py` / `kv_cache_manager.py` by session** (whole sessions, longest expected time away first). Oracle mode: the harness passes the true pause in `vllm_xargs`. **Continuum baseline:** TTL pinning (pinned sessions' blocks held out of the free queue until expiry) and its queue order (try vLLM's priority scheduling with priority = session arrival before writing scheduler code). Test with 0.5B first. | Ours, Belady and Continuum running in vLLM. |
| Sat | Simulator vs real for always-offload and ours at matching load and cap. | **Gate D:** within 15% → tune in the simulator; otherwise tune on real runs and report the gap. |
| Sun | **Gate E (hardware):** with calibrated numbers, does the simulator show an eviction gain on the 2080 Tis at any load and cap? If not, choose a memory-bound setup for week 3: (a) a smaller model (Qwen2.5-Coder-1.5B or 3B) with a squeezed KV cap, so compute is fast relative to memory; or (b) rent a faster GPU (A100/H100 class) for the sweep (needs a cost estimate and the user's approval). | Setup for week 3 decided. |

## Week 3 (Nov 2–8): experiments, figures, write-up

| Day | Work | Output |
|---|---|---|
| Mon | Set up the Gate E choice; find the memory cliff there (load sweep under always-offload); pick three loads around it. Sweep driver that restarts the server between runs and writes `results/<config>/<load>/<cap>/<run>/`. | Load levels and caps fixed. |
| Tue night → Thu | Full sweep: 6 configs (vLLM default, always-offload, Continuum, ours, ours + Continuum's queue order, Belady) × 3 loads × 2 caps × 3 runs = 108 runs × 12 min ≈ 22 GPU-hours, sequential. | All runs done, logs archived. |
| Fri | Figures on gpubox: goodput vs load per cap; returning-turn TTFT p50/p99; **new sessions' first-token wait** (the queue order's cost); share of returns found on GPU / CPU / gone; time-left guess quality; simulator vs real. | `results/figures/` |
| Sat–Sun | Check results, rerun noisy cells, start the write-up. Framing: remaining-time eviction is our contribution; Continuum's gain is mostly scheduling; the two combine. | Draft. |

## Slack and cuts

- **Most likely slip:** C1 eviction order in vLLM (week 2 Wed–Fri). Cut order if it slips: Continuum's
  TTL pinning in vLLM first (keep its queue order and the simulator comparison), then the
  "ours + queue order" config.
- **Biggest risk:** Gate E. If neither option gives a memory-bound regime, the paper rests on the
  simulator plus a real validation of its regime boundary on the 2080 Tis.
- **Stretch goals, only with time left:** prefetch (C4), noisy oracle (±50%), a better time-left guess
  from command arguments.
- **Parallel work:** simulator studies run on CPU while GPU runs are going.
- **Commit after each passed gate**, per the CLAUDE.md workflow.
