# TTFT improvement plan (Oct 27 – Nov 1, 2026)

Five ways to cut time to first token, ranked by expected gain (discussion of Oct 27). This plan
tests the four that need hardware or the simulator, in an order where each result feeds the next.
It fits into week 2 of `plan.md` and replaces its Mon–Tue rows (the C6 harness is not started yet).

| # | Idea | Evidence so far | Expected gain |
|---|---|---|---|
| A | Cut the CPU-reload overhead | 8K reload = 0.95 s TTFT at TP=2, but the raw copy is 10–35 ms | every returning turn that misses the GPU; may revive prefetch |
| B | Faster attention kernel on Turing | kernels ~8–10× below hardware at long context; 1K new tokens on a cached 16K context = 3.5 s | every request |
| C | Queue order: serve by session arrival | simulator: +12% to +34% at 1× (tight cap), +30% to +70% with our eviction at 4× | largest; costs new sessions some wait |
| D | Chunked-prefill tuning | none yet | returning turns stuck behind long cold prompts |
| E | Better time-left guess | ours gets 1/3–1/2 of Belady's gain | up to +34% at the memory cliff (simulated) |

## Step 0 (Tue–Wed): C6 replay harness and metrics — needed by A–D

As in `plan.md` week 2 Mon: asyncio + httpx, recorded prompts, `min_tokens = max_tokens` forced,
recorded tool sleeps, Poisson session arrivals, session ID in `vllm_xargs`. Two additions for this plan:

- an optional `priority` field per request (the OpenAI endpoint takes `priority: int`, lower = sooner),
  so step 2 needs no harness changes;
- per-request log of TTFT, new-token count, context length, and whether the turn is a session's first.

Exit: GPU test passes (replayed prompts match the trace token for token); a 10-minute run at
6/min writes per-request records under `results/`.

## Step 1 (Wed–Thu): measure A and B — single requests, no harness needed

Small scripts like `scripts/gate_b_warmup.py`, run while the harness is being finished. Every number:
warm-up first, 5 repeats, report median and spread.

### A. Why is reload slow?

1. Reload TTFT at 2K / 4K / 8K / 16K context, TP=2 and TP=1 (`GPU=1 TP=1`), with LMCache logging on.
   Separate the time spent in the LMCache lookup, the copy, and the forward pass over the
   last few uncached tokens.
2. LMCache settings: `LMCACHE_CHUNK_SIZE` 256 vs 1024; layer-by-layer loading (`LMCACHE_USE_LAYERWISE`)
   and async loading if 0.5.6 supports them on our path. Check the LMCache 0.5.6 docs for exact names first.
3. **Stall test:** one long decode stream running; send a reload request; measure the decode stream's
   time between tokens before, during and after. This answers the open question in `plan.md` week 2 Tue.
4. If a torch profiler trace is needed, take one at 8K, TP=2.

Exit: reload TTFT at 8K/TP=2 explained, and ideally ≤ 0.3 s. Write the reload curve and stall cost
into the calibration JSON (`plan.md` week 2 Tue).
**Decision:** if reloads become cheap *and* don't stall the batch, re-run prefetch in the simulator
(it lost only because each warm-up blocked a step for 0.25 s). Otherwise prefetch stays out.

### B. Attention kernel

Candidates on sm_75: `TRITON_ATTN` (current), `FLEX_ATTENTION`, and `FLASHINFER`. FlashInfer is
switched off for sm_75 in our vLLM (`flashinfer.py:514`, upstream bug flashinfer#3620, fixed in
PR #3621): test it only if our installed FlashInfer has that fix, by relaxing the check in our fork.

For each backend (`ATTN_BACKEND=... scripts/run_vllm.sh`, 7B-AWQ, TP=2):
1. Cold prefill at 2K / 8K / 16K.
2. **Returning-turn case:** 1K new tokens on a cached 8K and 16K context (the Gate B case, 3.5 s now).
3. Decode step time at batch 1 / 8 / 32.
4. Correctness: greedy outputs on 20 recorded prompts match `TRITON_ATTN` (or differ only by fp16 noise).

Exit: a table per backend. **Decision:** switch the default if one is ≥ 20% faster on case 2 with
correct output. Then redo the calibration with it. A faster kernel makes the GPUs less compute-bound,
which helps Gate E.

## Step 2 (Thu–Fri): C queue order, and D chunk size

### C. Queue order by session arrival

1. **Check the shortcut:** run vLLM with `--scheduling-policy priority`; the harness sends
   `priority = ms since run start when the session arrived`. Our vLLM already preempts the
   lowest-priority running request when blocks run out (`scheduler.py:769`).
2. In the simulator, add a "session FCFS only" variant (Continuum's order without "pinned first") and
   compare with the full `remaining-evict-fcfs`. If they match, the vLLM shortcut is enough and no
   scheduler code is needed.
3. On hardware, always-offload with FCFS vs with priority, loads 6 and 8/min at the tight cap, 2 runs
   each (~1.5 GPU-hours). Report returning-turn goodput, all-request goodput, and **new sessions'
   first-token wait**.

Exit: the hardware gain points the same way as the simulator (+12% at 6/min, +34% at 8/min).
**Decision:** if yes, "ours + queue order" in week 3 uses the built-in priority path.

### D. Chunked-prefill size

On the best backend from B, at 8/min tight cap, one run each with `--max-num-batched-tokens`
512 / 2048 (default) / 8192. Look at returning-turn TTFT p50/p99 and decode time between tokens.

Exit: pick a value for all week 3 runs; then it is a fixed setting, not a result.

## In parallel on CPU (any day): E better time-left guess

In the simulator (`src/brb/predictor.py`, `remaining-evict`):
1. Add the command's target (test file, directory, build target) as a key above session+tool in the
   back-off chain.
2. Report guess error (median absolute error of time left) and goodput at 4×, 22 and 24/min, 5 seeds.

Exit: ≥ 2/3 of Belady's gain at 4×, 22/min, and no seed below always-offload. Otherwise keep the
current guess.

## What this changes in `plan.md`

- Week 2 Mon–Tue (harness + calibration) moves to Tue–Thu; calibration now includes A and B.
- C1 in vLLM (Wed–Fri) shrinks if C's shortcut works: no queue-order code, only eviction order.
- Most likely slip is still C1. If it slips past Sat, Gate D moves to Mon Nov 2 and the week 3 sweep
  drops to 2 runs per cell.
- Stays out: early offload, partial-session eviction (both lost or were ruled out in week 1).
