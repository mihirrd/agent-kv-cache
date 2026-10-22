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

## Gate C in the simulator (Oct 22): FAILS with placeholder calibration

Simulator `kvtier/sim` (commit a9dc30b), training split (56 traces), placeholder calibration
`GATE_A_TP2` (reload overhead 0.25 s and blocking; decode 25 ms + 0.5 ms/request). Goodput =
returning turns with TTFT ≤ 5 s per second. Mean of 3 seeds; results in gpubox `results/sim/`.

**Load range (vLLM default, full 206K pool).** The server saturates at ~8 sessions/min. As load
rises, running requests take the pool and paused-session KV shrinks:

| Load (/min) | Returning ≤ 5 s | KV in use (running) | KV held by paused sessions |
|---|---|---|---|
| 4 | 100% | 35K | 109K |
| 6 | 98% | 72K | 86K |
| 8 | 83% | 123K | 46K |
| 10 | 70% | 153K | 25K |
| 12 | 55% | 171K | 13K |
| 16 | 36% | 186K | 4K |

The agreed tight-cap rule (half the paused KV at the middle load) gives 12.5K tokens at 10/min, below
the 153K the running requests need, so it is infeasible. The runs below used loads 4, 6 and 8 with the
cap per load = KV in use + half the paused KV (89K, 115K, 146K). **This changes an agreed decision and
needs the user's sign-off.**

**Goodput relative to always-offload at the tight cap:**

| Policy | Load 4 | Load 6 | Load 8 |
|---|---|---|---|
| always-offload (abs.) | 0.489 | 0.551 | 0.451 |
| vLLM default (no CPU tier) | −19% | −17% | −15% |
| oracle (C3 rules + prefetch) | −10% | −8% | −8% |
| oracle-evict (eviction order only) | −2% | −0% | −2% |
| predictive (warm start) | −44% | −61% | −49% |
| predictive-evict | −42% | −63% | −52% |

Fallbacks (compiled-language traces only, caps 60K–100K, loads 4 and 6): oracle −2% to −17%,
oracle-evict −4% to +2%. **No cell reaches the +5% bar.**

What the numbers show:
1. **The CPU tier is worth 15–20%** over vLLM default (up to 60% on build-heavy traces at tight caps).
2. **Knowing pause lengths adds nothing to eviction order.** LRU evicts the session paused longest;
   with heavy-tailed pauses that session is very likely still in a long pause, so LRU picks nearly
   the oracle's victim.
3. **Proactive offload + warm-up prefetch costs more than it saves.** Each warm-up blocks a step for
   the 0.25 s reload overhead and re-prefills up to 255 tokens.
4. **The predictive policy's "low-confidence first, by LRU" eviction rule is harmful.** It evicts
   sessions paused milliseconds ago, which return at once (30–58% of returns come from CPU vs 8–43%).
5. **The server is compute-bound** where memory pressure exists; returning TTFT is dominated by
   queueing and prefill of new tokens.

**Long-call offload (user's idea, Oct 22).** Classify each tool call as long or short when its pause starts
(predicted p50 > T, or the true pause for an upper bound); offload long ones at once, keep short ones;
LRU eviction; no prefetch. Thresholds T = 1, 2, 5, 10 s; same loads and caps; 3 seeds.

| Load, cap | Guess T=1 | T=5 | T=10 | Perfect T=1 | T=5 | T=10 |
|---|---|---|---|---|---|---|
| 4, tight | −5.0% | −0.2% | −0.7% | −5.4% | −3.0% | −1.6% |
| 6, tight | −3.1% | +0.1% | −0.2% | −5.3% | −1.5% | −0.7% |
| 8, tight | −4.9% | −2.2% | −0.6% | −7.5% | −2.4% | −2.9% |
| 6, full | −0.9% | +0.5% | +0.6% | −4.0% | −1.0% | +0.1% |
| 8, full | −8.1% | −2.8% | −1.6% | −11.2% | −5.3% | −4.7% |

Guess quality at load 6, tight cap: for T = 1 s, 21% of pauses are long; precision 86%, recall 46%.
For T = 10 s, 5% are long; precision 50%, recall 12%.

More offloading is consistently worse, even with a perfect guess. Every finished turn is already
stored to CPU and a paused session's GPU blocks are reclaimed instantly when needed, so offloading
early frees memory nobody needs yet. LRU frees the same long-paused session under pressure, while
early offload makes returns that could have hit GPU pay a reload. Tool-awareness can only pay through
timing (prefetch hiding the reload) or by avoiding work (not storing short calls).

Caveats before treating this as final:
- The calibration is placeholder. The verdict depends on reload overhead and on whether reloads
  block the step; an asynchronous prefetch path (LMCache controller API) would make proactive
  offload cheaper.
- The simulator is not yet validated against real runs (Gate D).
- TTL-pin as implemented equals LRU (a session's last use equals its pause start). A real pin must
  refuse to evict pinned sessions.

## Implementation status (end of Oct 20)

| Component | Status |
|---|---|
| `scripts/run_vllm.sh` | TP=2 default, LMCache CPU offload, `PYTHONHASHSEED=0` |
| C1 hook | scheduler logs `vllm_xargs` session ID (`KVTIER_LOG_SESSIONS=1`) |
| C2 `kvtier/predictor.py` | back-off keys, ring buffer 50, p10/p50/p90, confidence rule; tested |
| C3 `kvtier/cost.py` | interpolated prefill cost, KEEP/OFFLOAD/DISCARD, eviction score, prefetch time; placeholder calibration from Gate A; tested |
| `kvtier/policy.py` | `KVPolicy` interface shared by the simulator and vLLM; methods take `now` |
| C5 `kvtier/recorder.py` | done; 70 traces recorded (50 Verified, 20 Multilingual compiled) |
| C7 simulator | next |
| Gate C (headroom) | after the simulator, on an 80% training split of traces |
