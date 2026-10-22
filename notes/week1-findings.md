# Week 1 findings (Oct 19–25, 2026)

Measurements and decisions from week 1, for reference when writing the report. All numbers are from
gpubox (2× RTX 2080 Ti, 11 GB each, NVLink NV2). Unless noted, each number is a single sample.

## Stack

| Component | Version / setting | Why |
|---|---|---|
| NVIDIA driver | 580.126.09 (CUDA 13.0), graphics-drivers PPA | LMCache ≥ 0.4.5 needs CUDA 13; the old 535 driver topped out at CUDA 12.2 |
| Python | 3.11 via `uv python install` | system python3 is 3.8 |
| vLLM | 0.31.0, submodule `third_party/vllm`, branch `kvtier` | last stable release before LMCache 0.5.6 |
| LMCache | 0.5.6 | latest; wheel built for torch 2.13 / CUDA 13, includes sm_75 |
| torch | 2.13.0 (CUDA 13 build from PyPI) | pinned by vLLM 0.31.0 |
| Attention backend | `TRITON_ATTN` | only `TRITON_ATTN` and `FLEX_ATTENTION` are offered on Turing |
| Model | Qwen2.5-Coder-7B-Instruct-AWQ, fp16, TP=2 | see "TP=2" below |
| Docker | 28.1.1 (last release for Ubuntu 20.04) | SWE-bench task containers |
| Agent for recording | mini-SWE-agent 2.4.6 + Claude Haiku 5.5 via LiteLLM | |

Gotchas:
- The editable vLLM install puts kernel `.so` files into `third_party/vllm/vllm/` on gpubox;
  `scripts/remote.sh` protects that tree from rsync `--delete`.
- With TP > 1, LMCache misses every CPU lookup unless `PYTHONHASHSEED` is fixed. The scheduler and
  workers are separate processes, and LMCache keys chunks with Python's `hash()`.
  `run_vllm.sh` sets `PYTHONHASHSEED=0`.
- The tmux server predates the `docker` group, so Docker commands inside tmux need `sg docker -c '...'`.

## Gate A: vLLM + LMCache serve on sm_75 (PASS)

KV cache size: 28 layers × 4 KV heads × 128 dim × (K, V) × 2 bytes = **57,344 bytes/token**
(about 0.9 GB for a 16K-token session).

### Prefill and reload, 7B-AWQ

Time to first token with `max_tokens=1`. "Cold" means no cached KV anywhere.

| Measurement | 1 GPU, TRITON_ATTN | 1 GPU, FLEX_ATTENTION | TP=2, TRITON_ATTN |
|---|---|---|---|
| GPU KV pool | 36,384 tokens | 35,648 tokens | **205,936 tokens** |
| 8K cold prefill | 11.9 s (675 tok/s) | 13.3 s | 7.6 s (1,055 tok/s) |
| 16K cold prefill | 49.6 s (323 tok/s) | 54.8 s | 27.9–29.2 s |
| 28K cold prefill | 156.8 s (179 tok/s) | 168.3 s | 81.9–99.2 s |
| 8K, KV still on GPU | 0.14 s | 0.56 s | 0.09–0.17 s |
| 8K, KV reloaded from LMCache CPU | 0.27 s | 0.77 s | 0.95 s (one sample) |

- **Prefill slows roughly quadratically with prompt length** on Turing: 2× the tokens costs ~4×
  the time. Hardware estimates put a 28K prefill at ~15–20 s, so the Triton kernels are ~8–10×
  slower than the GPU should allow. FlashAttention needs sm_80+.
- **LMCache transfers are fast:** storing to CPU runs at 3–6 GB/s and loading back at 10–24 GB/s
  (8K tokens ≈ 0.2–0.4 GB). Most of the reload TTFT is overhead, not the transfer.
- **The TP=2 CPU-reload TTFT of 0.95 s (vs 0.27 s on one GPU) is unexplained.** The per-GPU copies
  took 10 ms and 35 ms. Likely causes are a one-time Triton JIT compile or cross-worker coordination.
  Re-measure in week 2 calibration with warm-up and repeats.
- Run-to-run noise is visible (28K at TP=2: 82 s, then 99 s). GPU 0 also drives the desktop.

### Decisions
- **TRITON_ATTN over FLEX_ATTENTION.** FLEX was 5–15% slower on prefill and 2–4× slower on cache
  hits; its per-step block-mask overhead (~0.4 s) masks the GPU-hit vs CPU-reload gap the research measures.
- **TP=2 as the standard setup.** It gives 5.7× the KV pool and ~1.9× faster prefill, with
  all-reduce over NVLink. Cost: no parallel sweeps on the two GPUs, so the full sweep runs sequentially (~2 nights).
- **Cap agent contexts at ~16K tokens.** 30K contexts don't fit the run budget at these prefill speeds.

## Gate B: warm-up request gives a GPU prefix-cache hit (PASS)

`scripts/gate_b_warmup.py`, 7B-AWQ TP=2, chat API with Qwen's chat template:

| Step | Prompt tokens | Time | GPU hits | LMCache hits |
|---|---|---|---|---|
| A turn 1 (cold) | 16,069 | 28.4 s | 0 | 0 |
| *240K tokens of filler evict A and B from GPU* | | | | |
| A warm-up (last prompt, `max_tokens=1`) | 16,069 | 0.78 s | 0 | 15,872 |
| *40K tokens of other traffic* | | | | |
| **A turn 2 (after warm-up)** | 17,064 | **3.56 s** | **16,064** | **0** |
| B turn 2 (no warm-up, control) | 17,056 | 4.26 s | 32 | 15,840 |

- A warm-up turns the next real turn into a GPU prefix-cache hit through the chat template, and
  the warmed blocks survive 40K tokens of other traffic. Prefetch (C4) stays in scope.
- The saving is the CPU-reload cost (~0.7 s here). The remaining ~3.5 s is prefill of ~1,000 new
  tool-output tokens attending over a 16K context, which no KV policy can avoid.

### Session ID path
- **`vllm_xargs`** (OpenAI SDK: `extra_body={"vllm_xargs": {"session_id": ...}}`) reaches the
  scheduler in the EngineCore process. Verified with a `KVTIER_LOG_SESSIONS=1` log line for raw
  chat, `/v1/completions`, and SDK streaming (the LiteLLM path).
- `X-Request-Id` also reaches the engine, as the request ID prefix.
- The OpenAI `user` field is **not** forwarded to the engine.

## C5 trace recording

### Pilot 1: plain mini-SWE-agent prompt (10 SWE-bench Verified tasks)
- All 10 submitted in 3–12 turns (median 5) for $0.04 in total.
- Haiku reads, edits with `sed` and submits without testing. `pytest` ran once in 62 tool calls.
- **Tool pauses: p50 0.05 s, p90 0.26 s, max 4.8 s.** Too short for KV placement to matter.

### Pilot 2: with `TESTING_WORKFLOW` (same 10 tasks)
The task prompt now requires the agent to: reproduce the bug, re-run the reproduction after the
fix, and run the repo's existing tests for changed modules before submitting. The per-command
timeout went from 60 s to 600 s.
- Turns: median 9. Cost: $0.035.
- **Pauses: p50 0.10 s, p90 2.6 s, max 26 s.** The tail grew ~10×.

### Full Verified set: 50 tasks, `traces/raw/verified/`

| Metric | Value |
|---|---|
| Outcome | 49 submitted, 1 hit the 15.5K context cap |
| Cost | $0.262 total, $0.005/task |
| Turns/session | min 5, p50 10, p90 16, max 20 (548 total) |
| Final context (Qwen tokens) | min 3.6K, p50 6.2K, p90 8.7K, max 14.7K |
| Haiku request time | p50 1.9 s, p90 5.5 s |
| Output tokens/turn | p50 288, p90 1,101 |

Tool pauses, 539 tool calls:

| p10 | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 0.04 s | 0.09 s | 3.2 s | 20.9 s | 65.4 s |

| Pauses longer than | Share of pauses | Share of total pause time |
|---|---|---|
| 1 s | 22.8% | 92.0% |
| 5 s | 6.7% | 66.4% |
| 10 s | 3.2% | 49.1% |
| 30 s | 0.7% | 22.7% |

| Category | n | p10 | p50 | p90 | max | Main programs |
|---|---|---|---|---|---|---|
| inspect | 304 | 0.04 s | 0.05 s | 0.46 s | 20.6 s | git, grep, sed, echo |
| run | 141 | 0.07 s | 0.24 s | 1.71 s | 9.3 s | python |
| install | 54 | 1.70 s | 3.36 s | 17.98 s | 65.4 s | pip |
| other | 23 | 0.21 s | 0.80 s | 2.79 s | 21.1 s | `$PY`, rm, mkdir |
| test | 17 | 0.06 s | 2.05 s | 13.77 s | 32.8 s | pytest, runtests.py |

What this means:
1. **Idle KV time is concentrated in a few long pauses.** The 7% of pauses over 5 s account for
   two-thirds of all idle KV time. This is the case tool-aware placement targets.
2. **The tool category predicts pause length.** `inspect` and `install` meet the confidence rule
   (p90/p10 ≈ 11). `test` does not (p90/p10 ≈ 230): one quick test and a full suite look the same
   by program name. The test target may be a better key.
3. **Memory pressure needs many concurrent sessions.** Sessions peak at ~6–9K tokens
   (0.35–0.5 GB of KV), so ~25–35 fit in the 206K-token TP=2 pool. Replay needs high concurrency
   or a lower `--gpu-memory-utilization`.
4. `pip` is ~10% of tool calls: the agent installing test tooling in fresh containers.

### Multilingual compiled set: 20 tasks, `traces/raw/multilingual/`
Rust, Go, Java and C/C++ repos from SWE-bench Multilingual (`--dataset multilingual --compiled`):
tokio, axum, bat, ruff, nushell, caddy, gin, hugo, terraform, lombok, gson, lucene, druid, redis, nlohmann/json.

| Metric | Value |
|---|---|
| Outcome | 19 submitted, 1 hit the context cap (hugo-12562, 20 turns) |
| Cost | $0.115 total, $0.006/task |
| Turns/session | min 4, p50 9, p90 18, max 24 (208 total) |
| Final context (Qwen tokens) | min 4.1K, p50 6.7K, p90 14.2K, max 15.2K |

Tool pauses, 203 tool calls:

| p10 | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 0.04 s | 0.06 s | **12.1 s** | **72.8 s** | **126.3 s** |

| Pauses longer than | Share of pauses | Share of total pause time |
|---|---|---|
| 1 s | 25.6% | 97.9% |
| 5 s | 14.3% | 92.6% |
| 10 s | 10.3% | 86.8% |
| 30 s | 4.9% | 62.6% |

| Category | n | p10 | p50 | p90 | max | Main programs |
|---|---|---|---|---|---|---|
| inspect | 157 | 0.04 s | 0.05 s | 1.08 s | 73.2 s | sed, grep, cat, git |
| run | 22 | 0.06 s | 3.54 s | 25.9 s | 34.2 s | python3, perl |
| other | 13 | 0.06 s | 5.92 s | 94.4 s | 126.3 s | rm, mkdir, printf, ant |
| build | 11 | 0.82 s | **21.3 s** | 42.7 s | 45.0 s | mvn, go, cargo, gradlew |

- The compiled set roughly doubles the tail of the Python set: p90 12 s vs 3 s, and 10% of pauses over 10 s vs 3%.
- Build and test steps (`build`, p50 21 s) are the long, predictable pauses the policy targets, but
  they are rare: 11 of 203 calls. Agents mostly inspect and edit.
- `ant` (Java builds) was missing from the `build` category; added after this run.

### All 70 traces combined

| Metric | Value |
|---|---|
| Outcome | 68 submitted, 2 context cap; $0.377 total |
| Turns/session | p50 10, p90 16, max 24 (756 total) |
| Final context | p50 6.3K, p90 9.4K, max 15.2K Qwen tokens |
| Tool pauses (742) | p10 0.04 s, p50 0.08 s, p90 4.4 s, p99 39.5 s, max 126 s |
| Pauses > 5 s | 8.8% of pauses, **80.5% of total pause time** |
| Pauses > 30 s | 1.9% of pauses, 44.2% of total pause time |

Docker images for the 70 tasks use ~60 GB; 282 GB free afterwards.

## Oct 22: simulator, Gate C, and a new direction

Simulator `kvtier/sim` on the 56-trace training split, placeholder calibration `GATE_A_TP2` (reload
overhead 0.25 s that blocks the batch; decode 25 ms + 0.5 ms per request). "Goodput" = requests per
second whose time to first token is ≤ 5 s. All results are simulated; raw outputs in gpubox
`results/sim/`.

### In short

1. **On our 2080 Tis the server runs out of compute before memory.** Paused sessions only crowd the
   GPU at loads it can already barely serve, so memory policies have little to act on.
2. **Moving memory early does not pay:** proactive offload, warm-up prefetch and "offload the long
   calls" all lose to always-offload, even with perfect knowledge of pause lengths.
3. **When memory is the bottleneck (simulated faster GPU), choosing *whom to evict* pays.** Evicting
   the session expected to stay away longest gains up to +11% and never loses; a perfect guesser
   gains up to +34%.
4. **Continuum's gain comes from its queue order, not its memory reservation.** As a memory policy,
   ours beats its TTL pinning; combined with its queue order, ours is best on fast GPUs.

### Decisions

| Decision | Why | Status |
|---|---|---|
| Headline SLO: 5 s TTFT; also report 2 s and 10 s | a fully cached 16K turn still takes ~3.5 s (Gate B) | agreed |
| Always-offload baseline keeps KV on GPU until pressure, reloads from the LMCache copy | LMCache's default; the stronger baseline | agreed |
| Headline: returning-turn goodput, all-request goodput beside it | returning turns are what a KV policy can help; all-request keeps it honest | agreed |
| Predictor reported warm-started (main) and cold | a deployed system has history | agreed |
| Policies evict whole sessions | a partial prefix is mostly useless; Gate D will measure the gap to vLLM | agreed |
| Tight cap = KV in use + half the paused KV, per load (89K / 115K / 146K at 4 / 6 / 8 per min) | the agreed rule (half the paused KV at the middle load) gives 12.5K tokens, below what running requests need | **pending user sign-off** |
| Prefetch set aside for now | user's call; warm-ups also lost in every test | agreed |
| "Offload long calls early" rejected | loses even with a perfect long/short guess | from the data |
| **New direction: treat it as a caching policy.** Keep everything while there is room; when space runs out, evict the session expected to stay away longest | the gain is in knowing *what to evict*, not what to keep (user's direction) | adopted |
| Simulate faster GPUs (`--speedup`) as a stand-in for memory-bound hardware | our GPUs are compute-bound | adopted |
| Compare against Continuum (arXiv:2511.02230) in the simulator | closest prior work; it was already the spec's baseline | done |

### Finding 1: on today's GPUs, compute runs out first

Load sweep, vLLM default, full 206K pool. As load rises, running requests take the pool and paused
sessions get squeezed out:

| Load (/min) | Returning ≤ 5 s | KV in use (running) | KV held by paused sessions |
|---|---|---|---|
| 4 | 100% | 35K | 109K |
| 6 | 98% | 72K | 86K |
| 8 | 83% | 123K | 46K |
| 10 | 70% | 153K | 25K |
| 12 | 55% | 171K | 13K |
| 16 | 36% | 186K | 4K |

Gate C as specified (oracle beats always-offload by ≥ 5%) **fails** at the tight caps:

| Policy | Load 4, 89K | Load 6, 115K | Load 8, 146K |
|---|---|---|---|
| always-offload (absolute) | 0.489 | 0.551 | 0.451 |
| vLLM default (no CPU copy) | −19% | −17% | −15% |
| oracle (C3 rules + prefetch) | −10% | −8% | −8% |
| oracle, eviction order only | −2% | −0% | −2% |
| predictive (warm start) | −44% | −61% | −49% |

The fallbacks (build-heavy traces only, caps 60K–100K) also stay between −17% and +2%.

- The CPU copy (LMCache) alone is worth 15–20% over vLLM default.
- LRU already picks nearly the oracle's victim. With heavy-tailed pauses, the session paused longest
  is almost always in a long pause.
- Each warm-up blocks a step for the 0.25 s reload and re-prefills up to 255 tokens, so prefetch costs
  more than it saves.

### Finding 2: offloading long calls early does not help

Classify each tool call as long or short when its pause starts (predicted median > T, or the true
pause for an upper bound); offload long ones immediately; LRU otherwise; no prefetch. 3 seeds.

| Load, cap | Guess T=1 s | T=5 s | T=10 s | Perfect T=1 s | T=5 s | T=10 s |
|---|---|---|---|---|---|---|
| 4, tight | −5.0% | −0.2% | −0.7% | −5.4% | −3.0% | −1.6% |
| 6, tight | −3.1% | +0.1% | −0.2% | −5.3% | −1.5% | −0.7% |
| 8, tight | −4.9% | −2.2% | −0.6% | −7.5% | −2.4% | −2.9% |
| 6, full | −0.9% | +0.5% | +0.6% | −4.0% | −1.0% | +0.1% |
| 8, full | −8.1% | −2.8% | −1.6% | −11.2% | −5.3% | −4.7% |

Guess quality (load 6, tight cap): for T = 1 s, 21% of pauses are long, precision 86%, recall 46%;
for T = 10 s, 5% are long, precision 50%, recall 12%.

Why it loses: every finished turn is already copied to CPU, and a paused session's GPU memory is
handed to a new request the moment it is needed. Offloading early frees memory nobody needs yet, and
turns returns that could have hit GPU into reloads.

### Finding 3: when memory is the bottleneck, eviction order matters

`--speedup S` makes prefill and decode S× faster with the same 206K-token pool and the same reload
cost. With 4× compute the pool fills with *paused* KV (176K tokens at 8/min) while every request is
on time; the cliff where memory runs out is between 16/min and 24/min. With 8×, between 24 and 32/min.

Eviction-only oracle vs always-offload: +14.6% at 4×, 24/min (seeds +11% to +17%); +5.4% at 8×,
32/min. Proactive offload + prefetch loses badly here (−46% to −86%), because the fixed reload now
costs several batches. The first predictive rule (C3 score, "low-confidence first") lost 68–99%: it
treated overdue sessions as about to return and never evicted them.

### Finding 4: remaining-time eviction (our caching policy)

Keep everything while there is room. When space runs out, evict the paused session expected to stay
away longest. Expected time left = median of (x − elapsed) over past pauses x of the same tool that
lasted longer than this pause so far (back-off: session+tool → tool → category → all; infinite if
none lasted that long). Returns reload from the LMCache copy. Full pool, 5 seeds, returning-turn
goodput vs always-offload:

| Policy | 4×, 18/min | 4×, 20 | 4×, 22 | 4×, 24 | 8×, 30 | 8×, 32 |
|---|---|---|---|---|---|---|
| always-offload (absolute) | 2.87/s | 3.02/s | 2.30/s | 1.61/s | 4.96/s | 5.09/s |
| **ours** | +0.3% | +0.9% | **+11.4% [+1, +30]** | **+7.4% [+1, +15]** | +0.1% | +1.5% [0, +7] |
| perfect (Belady: evict who returns last) | +0.3% | +1.4% | +33.9% [+1, +74] | +12.9% [+4, +22] | +0.3% | +3.5% [0, +16] |
| vLLM default | −9% | −45% | −46% | −41% | −25% | −51% |

- The first tool-aware policy that never loses: its worst seed is ≥ 0% in every cell.
- It captures a third (22/min) to half (24/min) of the perfect guesser's gain. At 4×, 22/min it keeps
  79% of returning sessions on GPU, against 74% for LRU and 90% for Belady.
- Gains sit at the memory cliff; below it nothing is evicted, above it the GPU is overloaded anyway.
- On today's GPU with squeezed memory it ties with always-offload (compute-bound).

### Finding 5: head-to-head with Continuum

**What Continuum does** ("Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV
Cache Time-to-Live", Li et al., arXiv:2511.02230; vLLM 0.10.2 + LMCache 0.3.7; A100/H100/B200;
mini-SWE-agent SWE-bench and BFCL traces):

1. **Reserve memory for a set time (TTL).** When an agent starts a tool call, its KV is pinned. The
   TTL is chosen once, at pause start: the τ that maximises P(pause ≤ τ) × (T·η + prefill − reload) − τ,
   where P comes from past durations of that tool (all tools if it has ≤ 100 records) and T is the
   average queueing delay. While pinned, nobody else may use that memory. If the queue is stuck,
   pinned sessions are evicted latest-arriving first.
2. **Change the queue order.** Serve requests by when their agent session first arrived, pinned
   sessions first. In effect it finishes started agents before taking on new ones.
3. It compares against vLLM with end-of-turn eviction and other research systems, not against LRU
   prefix caching or an oracle, and does not discuss compute- vs memory-bound regimes. No prefetch.

**How we differ:** we reserve nothing; we only choose whom to evict when space runs out, and we update
the guess as the pause goes on (an agent gone longer than usual is evicted first). We compare against
vLLM's LRU prefix cache with the LMCache copy and against a perfect guesser, and we leave the queue
order alone unless stated.

**Results** (our reimplementation; η = 1; same warm start for both; 5 seeds). All-request goodput at
5 s vs always-offload:

| Policy | 4×, 20/min | 4×, 22 | 4×, 24 | 8×, 32 | 1×, 6/min, 115K | 1×, 8/min, 146K |
|---|---|---|---|---|---|---|
| Continuum, pinning only (returning turns) | −0.1% | −2.5% | −0.4% | 0% | 0% | 0% |
| Continuum, full | +0.4% | +18.0% | +67.4% | +0.3% | +13.4% | +39.9% |
| ours | +0.8% | +11.3% | +6.8% | +1.5% | 0% | −0.2% |
| ours + Continuum's queue order | +2.0% | +30.2% | +70.5% | +2.6% | +11.8% | +34.1% |

Median wait for a new session's first token: at 4×, 24/min always-offload 10.2 s, Continuum 21.2 s,
ours + queue order 13.7 s; at 1×, 8/min 7.0 s, 21.9 s and 17.6 s.

1. **Continuum's gain comes from its queue order.** Pinning alone is 0 to −2.5%. The queue order also
   helps on today's compute-bound GPU, so it relieves compute congestion rather than memory.
2. **As a memory policy, ours beats pinning:** +7% to +11% vs ≤ 0% at 4×.
3. **Ours + their queue order is best on fast GPUs** (+30% / +71% at 4×, +2.6% at 8×) and level with
   Continuum on today's GPU with squeezed memory (+34% vs +40%; seed ranges overlap widely).
4. **The queue order makes new agents wait longer under overload.** Any write-up must report both.

Possible framing for the project: remaining-time eviction is our contribution; we show Continuum's
gain comes from scheduling rather than pinning; the two combine.

### Caveats and open items

- **Placeholder calibration.** Reload overhead (0.25 s, blocking) and decode speed are estimates; week
  2 calibration must measure them, including whether a reload stalls the batch.
- **Not validated.** The simulator has not been compared with real runs (Gate D).
- **Faster GPUs are simulated.** On our 2080 Tis memory does not run out before compute, so real runs
  here cannot show the eviction gain; that needs faster (rented) GPUs or a memory-squeezed setup.
- **Continuum rebuilt from the paper.** Its text was read through a paraphrasing tool; η fixed at 1.
- **Room to improve the guess:** ours gets a third to half of Belady's gain. Using the command
  (test target, directory) rather than only the program name is the obvious next step.
- **Superseded code:** `TtlPinPolicy` equals LRU (kept only for reference; `ContinuumPolicy` replaces
  it). The C3 `PredictivePolicy` eviction rule is broken (overdue sessions never evicted); remaining-time
  eviction replaces it.

## Implementation status (end of Oct 22)

| Component | Status |
|---|---|
| `scripts/run_vllm.sh` | TP=2 default, LMCache CPU offload, `PYTHONHASHSEED=0` |
| C1 hook | scheduler logs `vllm_xargs` session ID (`KVTIER_LOG_SESSIONS=1`) |
| C2 `kvtier/predictor.py` | back-off keys, ring buffer 50, p10/p50/p90, confidence rule, `remaining()` time-left estimate; tested |
| C3 `kvtier/cost.py` | interpolated prefill cost, KEEP/OFFLOAD/DISCARD, eviction score, prefetch lead; superseded for eviction by remaining-time eviction |
| `kvtier/policy.py` | `KVPolicy` interface shared by the simulator and vLLM; methods take `now` |
| C5 `kvtier/recorder.py` | 70 traces recorded (50 Verified, 20 Multilingual compiled) |
| `kvtier/sessions.py` | traces → replay sessions (Qwen block-hash chains), shared with C6 |
| C7 `kvtier/sim/` | done: LRU, always-offload, oracles, long-call offload, remaining-time eviction, Belady, Continuum; `--speedup`; 68 tests |
| Gate C | fails on today's GPUs; passes for eviction order in the simulated memory-bound regime |
| C6 harness, calibration, C1 eviction in vLLM, Gate D | week 2 |
