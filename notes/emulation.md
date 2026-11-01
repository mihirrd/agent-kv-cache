# Emulating a faster GPU on the 2080 Tis (default experiment setup, Oct 27)

The 2080 Tis are the only GPUs we have. With 7B-AWQ they run out of compute long before memory
(Oct 27 smoke run: overloaded at 6 sessions/min, paused KV a small share of memory), so a KV caching
policy has nothing to act on. This note sets up an environment where the policy *does* matter,
and argues why its results carry over to faster GPUs.

## Why results can transfer

In a replay the prompts are recorded and every reply's length is forced (`min_tokens = max_tokens`).
What the model writes never matters. The model is only a source of GPU work and KV memory. So what
decides whether a caching policy helps is not the model but three ratios:

| Ratio | Meaning | What a faster GPU does to it |
|---|---|---|
| **R1: tool time vs serving time** | share of an agent's life spent in tool calls (`pause_share`) | raises it: turns get served faster, tool calls take the same seconds |
| **R2: memory vs demand** | how many sessions' KV fit in the pool vs how many are alive (`kv_usage`, reloads) | set by pool size and load, which we choose |
| **R3: reload vs recompute** | cost of reloading a token's KV from CPU vs prefilling it again | depends on PCIe bandwidth, KV bytes per token and prefill speed |

If a run on our GPUs matches a target GPU on these three, the scheduler sees the same pattern of
arrivals, pauses, evictions and returns, and the policy should behave the same.

## The knobs

1. **A smaller model** (`PROFILE=emu`: Qwen2.5-Coder-0.5B fp16, one server per GPU). It uses the same
   tokenizer as 7B, so the traces replay unchanged. Serving is much faster (R1 up), and two servers
   run two experiments at once.
2. **Time stretching** (`--time-scale k` in the harness). *Optional, not part of the default setup*
   (user's call, Oct 27): use it only if the 0.5B's `pause_share` is too low, or for the 7B cross-check. Tool pauses and gaps between arrivals take k
   times longer on the wall clock, and all reported times are divided by k. To the scheduler this is
   a GPU k times faster. It is exact for compute; the one thing it also speeds up is reload, so R3
   moves too (see "What does not transfer"). Runs take k times longer.
3. **A fixed KV pool** (`NUM_GPU_BLOCKS`, 16 tokens per block). R2 is set in tokens, independent of
   how much memory the card has.

KV bytes per token (layers × KV heads × head dim × 2 × 2 bytes): 0.5B 12,288; 1.5B 28,672;
3B 36,864; 7B 57,344. R3 check (rough, to be measured): reloading one 0.5B token over our PCIe 3.0
(~12 GB/s) takes ~1 µs, about the same as one 7B token over an H100's PCIe 5.0 (~50 GB/s). So the
0.5B emulator keeps reload costs in a realistic range by construction.

**Why shrinking the pool alone is not enough with 7B:** in the 7B smoke run agents spent 5% of their
time in tool calls (16% at light load), so nearly all KV belongs to running requests. A small pool then
just starves running requests (queueing, preemption), which an eviction policy cannot influence. The
faster model raises `pause_share`, so paused sessions hold a real share of the pool.

## Oct 27–28 measurements: 0.5B capacity, and the real bottleneck

Full pool, seed 0. 0.5B handles ~10 sessions/min: at 10/min turns take 1.5–2.2 s and agents spend
34–47% of their time in tool calls (7B: 5%); paused KV averages ~24K tokens vs ~23K for running
requests. 12/min tips over midway (running 3 → 27); 15 and 20/min run away. **8/min also ran away
on this seed** (running 2 → 64, turns 20 s), so the tipping point depends on which sessions arrive
together: per-token cost grows with batch, so a burst makes everyone slower and sessions pile up.

Why: **attention for prompt chunks on Turing.** vLLM's Triton kernel (`triton_unified_attention.py`)
uses its fast split decode path ("3D") only when no prompt is in the step (`max_seqlen_q > 1` disables
it). Agent workloads almost always have a prompt chunk in flight. Kernel timings on GPU 1 (0.5B head
layout, which is also 7B's per-GPU layout at TP=2):

| per layer | replies only (3D) | + one 512-token prompt chunk |
|---|---|---|
| batch 8, 8K context | 0.39 ms | 6.3 ms |
| batch 64, 8K context | 2.9 ms | 8.7 ms |

So every mixed step costs ~150 ms of attention on the 0.5B (24 layers). The prompt chunk alone
costs 6–9 ms per layer, about 1% of the GPU's tensor-core throughput. PyTorch's fused SDPA kernel
("efficient", which uses tensor cores on sm_75) does the same chunk in 0.7 ms (4K context),
1.3 ms (8K) and 1.8 ms (16K). Decode-only attention also reaches only ~90 GB/s (15% of bandwidth).
FlashInfer (0.7.0.post1 installed) is disabled for sm_75 in our vLLM pending an upstream fix.

Consequence: until prompt attention is fixed, both 7B and 0.5B are limited by this kernel, not by
model size or memory. A modern GPU would not have this problem, so fixing it also makes the emulator
more faithful. Capacity and pool sizes above must be re-measured after any kernel change.

### Oct 29: SDPA for prompt chunks (vLLM fork `5440637`, `KVTIER_SDPA_PREFILL=1`, on by default)

Mixed steps now run decodes through the Triton 3D path and prompt chunks through PyTorch SDPA. Matches
the Triton kernel on random mixed batches (`tests/test_turing_attention.py`); 19/20 greedy replies
identical to stock (the other is fp16 rounding). A/B at 12/min, seed 0, 0.5B, full pool, run at the same
time on the two GPUs:

| | stock | SDPA prefill |
|---|---|---|
| returning TTFT p50 / p99 | 0.41 / 1.86 s | **0.18 / 1.61 s** |
| first-turn TTFT p50 | 0.49 s | 0.33 s |
| returning turn time p50 | 6.4 s | 5.1 s |
| running requests at end | ~25–30, climbing | ~34, climbing |

Time to first token improved 2.3×, but capacity barely moved (~10–12/min). The bottleneck is now
**decode attention**: the server reaches the prefill rate 12/min needs (~1,300–1,600 vs ~1,400 tok/s),
but decode throughput falls from ~390 to ~220 tok/s as running requests pass 30 (12/min needs ~380).
Decode reads every running request's full context each step at ~90 GB/s, ~15% of the card's bandwidth.

### Oct 30: GPU 1 overheats and throttles; results that used it are invalid

Sustained fp16 matmul for 60 s: GPU 0 holds 43.9 TFLOPS at 1350 MHz, 59 → 75°C, fan ~50%, no throttling.
GPU 1 starts at 32.6 TFLOPS, then reaches 89°C within ~10 s, `sw_thermal_slowdown` goes active, the clock
drops to 300 MHz and it settles at **6.0 TFLOPS** with the fan at 90%. It is ~7× slower than GPU 0
under load. Needs a physical fix (cooling/fan/airflow); until then use GPU 0 only.

Results that ran on GPU 1 (port 8001) are confounded and must not be used: the load-15 and load-20
probes, full-pool load 8 (sdpa), the SDPA side of the Oct 29 A/B (SDPA was handicapped, not helped),
and every small-pool collapse (load 8 @22.5K and @35K, load 10 @45/60/80K). **Every 7B TP=2 run is
also suspect**: TP=2 runs at the speed of the slower GPU, so the Oct 27 7B smoke run ("overloaded at
6/min") and possibly week-1 TP=2 timings (Gate A) were throttled. When GPU 1 started throttling is unknown.

Valid (GPU 0): load 10/12 probes, full-pool load 6 and 10 (both kernels), stock side of the A/B, and
small pools load 6 @21K (99% ≤ 2 s), load 10 @33.6K (87%, queue starting late in the run), load 8
@45K/60K/80K (100% ≤ 2 s; reloads 189K / 79K / 12K tokens).

### Nov 1: first policy comparison (GPU 0, load 8, 3 seeds per cell)

Our remaining-time eviction in vLLM (fork `b5f11c4`, `KVTIER_POLICY=remaining`) vs stock LRU vs the
oracle (true pause from the harness, `KVTIER_POLICY=oracle`), all with LMCache:

| pool | reloaded tokens: LRU | ours | oracle | gap closed |
|---|---|---|---|---|
| 45K | 282K | 192K | 121K | 56% |
| 60K | 109K | 38K | 15K | 76% |

TTFT is the same for all three (p50 0.066 s, 99.8–100% ≤ 2 s): a 0.5B reload costs milliseconds, so
fewer reloads do not show up in TTFT. Next: the same comparison without the CPU tier (a miss recomputes).
Runs before the orphan-block fix are in `results/replay/remaining-orphanbug/` and are invalid: the replay
never reuses the KV of a generated reply, and those blocks must be evicted before any session.

## Procedure for a new setting

1. **Find compute capacity:** replay at rising load with the full pool. Capacity is the highest load
   where `running_growth` stays under ~1.3 and `waiting_avg` is ~0.
2. **Pick loads below it** (about 50–80% of capacity), so compute keeps up. That is the regime a faster GPU is in.
3. **Shrink the pool** (`NUM_GPU_BLOCKS`) to the agreed tight-cap rule: KV of running requests + half the
   paused KV, time-averaged over a full-pool run at that load (estimate both from `requests.jsonl`).
   Check that paused sessions get evicted: `cpu_reloaded_tokens` > 0 and `returning_cached_share` drops,
   while `running_growth` stays low. **Do not use `kv_usage`:** vLLM counts only blocks held by running
   requests, not the cached blocks of paused sessions (at 15/min it peaked at 66% while LMCache
   reloaded 235K tokens).
4. **Record R1 to R3** in every summary (`pause_share`, `kv_usage_*`, `cpu_reloaded_tokens`,
   `returning_cached_share`; R3 from calibration).

## How we show it scales

Results are reported **against R1 (`pause_share`), not against a GPU name.** Any real GPU lands on
some point of that axis. The policy comparison is repeated at two or more points:

- 0.5B emulator at time scale 1, and at 2–4 (a still faster GPU);
- 7B at time scale k chosen to land on the same `pause_share` as an emulator run (slow, a few runs only).

**Pass:** at the same R1 and R2, the policy's gain over the baseline agrees within the run-to-run
noise across the two models. Then the gain depends on the ratios, not on our hardware, and the
claim for a faster GPU is: find its R1 to R3 and read off the gain.

## What does not transfer

- **Reload fixed overhead.** The 0.95 s 7B reload at TP=2 is mostly fixed cost, not bandwidth
  (unexplained). Time stretching divides it by k as if it were compute. Measure it per profile and
  report it beside R3.
- **Kernel quirks of Turing** (slow long-context attention). Contexts are capped at 16K and the model is
  small, which limits this, but absolute TTFT numbers are not claims about other GPUs; ratios and gains are.
- **Batch effects specific to a GPU** (how step time grows with batch size). The 7B cross-check is there
  to catch these.
