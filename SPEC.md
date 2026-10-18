Tool-Aware KV Cache Tiering: Technical Spec and Implementation Plan
Oct 16, 2026 · @Mihir Deshpande
Overview
We add a tool-aware KV cache manager to vLLM that predicts how long each agent pauses for a tool call and uses that prediction to place, evict, and prefetch the session's KV cache. The plan assumes 7 weeks (about Oct 19 to Dec 4, 2026), one NVIDIA GPU on the Purdue cluster, and a MacBook Air for GPU-free development.
In scope
	•	Session-aware tracking of paused agent requests in vLLM's V1 engine.
	•	An online pause-duration predictor keyed by session, command, and tool.
	•	A placement and eviction policy across GPU and CPU tiers, using LMCache for transfers.
	•	Predictive prefetch of offloaded KV before the agent returns.
	•	A trace recorder, a replay load generator, and a discrete-event simulator.
Out of scope
	•	Multi-GPU, tensor-parallel, or prefill/decode-disaggregated deployments.
	•	New CUDA kernels or changes to LMCache's transfer engine.
	•	Improving agent task success; only the workload shape matters.
	•	Disk tier, unless weeks 1 to 5 finish on schedule.
Success criteria
	•	Higher goodput than always-offload and fixed-TTL pinning at the highest tested load, at the same TTFT SLO.
	•	Never lower goodput than vLLM's default at any tested load.
	•	A predictor whose p10 to p90 interval contains the true pause in at least 75% of held-out cases.
	•	Simulator goodput within 15% of measured vLLM goodput for the same policy and load.
System architecture
Three new components sit inside vLLM (session table, predictor, policy) and one beside it (prefetch); the tooling on the left produces and replays the workload.

architecture · vLLM server components and experiment tooling
Lifecycle of one paused session
	•	A request finishes with a bash tool call; the session table records the session ID, context length, and program name, and marks the session PAUSED.
	•	The predictor returns p10, p50, and p90 for that program; the policy chooses KEEP, OFFLOAD, or DISCARD.
	•	Under memory pressure, the policy's eviction order decides which paused sessions lose their GPU blocks first; LMCache already holds a CPU copy.
	•	At pause start + p10 − reload time, the prefetch task sends a warm-up request that pulls the KV back into GPU blocks.
	•	The agent's next request arrives, hits the prefix cache, and the measured pause is fed back to the predictor.
Component specs
Eight components; the policy is written once against a shared interface so the simulator and vLLM run the same code.
class KVPolicy:     def on_pause(self, s: Session) -> Action: ...          # KEEP | OFFLOAD | DISCARD     def eviction_order(self, paused: list[Session]) -> list[Session]: ...  # first = evict first     def prefetch_at(self, s: Session) -> float | None: ...  # wall-clock time to start reload     def on_return(self, s: Session, pause_s: float) -> None: ...  # feeds the predictor
C1. Session tracking (vLLM, Python)
	•	The client sends a session ID in the OpenAI user field or an HTTP header; week 1 confirms which one reaches the engine.
	•	A SessionTable maps session ID to context length, KV block IDs, state (ACTIVE, PAUSED, OFFLOADED, DISCARDED), pause start time, and the current prediction.
	•	On request finish, parse the tool call from the output. mini-SWE-agent emits a bash code block in plain text, so the key is the program name in its last fenced bash block (pytest, grep, python).
	•	A returning request is matched by session ID and confirmed by a prefix-hash match on its first block.
C2. Pause predictor (pure Python)
	•	Back-off keys, most specific first: (session, program), (program), (tool), global.
	•	Each key holds a ring buffer of the last 50 observed pauses.
	•	Prediction = p10, p50, p90 from the first key with at least 5 samples.
	•	Confident when n ≥ 5 and p90 ÷ p10 ≤ 20; both thresholds are tunable.
	•	Label = arrival of turn N+1 minus finish of turn N, measured server-side.
C3. Cost model and placement policy
Inputs come from one calibration run: prefill rate (tokens/s), CPU-to-GPU bandwidth (GB/s), and KV bytes per token (128 KB for an 8B GQA model in fp16).
	•	Recompute cost = context tokens ÷ prefill rate.
	•	Reload cost = KV bytes ÷ bandwidth.
	•	On pause: KEEP if p90 is shorter than the offload round trip; DISCARD if recompute cost is below reload cost; otherwise OFFLOAD.
	•	Eviction score = recompute cost ÷ (KV GB × max(remaining p50, 0.1 s)). Lowest score is demoted first; sessions inside their prefetch window are exempt.
	•	Low confidence falls back to vLLM's LRU order for that session.
In vLLM this touches v1/core/block_pool.py (eviction order of freed, cached blocks), v1/core/kv_cache_manager.py, and the scheduler's request-finish path. File names are from recent vLLM releases; pin a version in week 1.
C4. Offload and prefetch (LMCache)
	•	LMCache stores each request's KV to CPU on completion, so OFFLOAD mostly means freeing the GPU blocks early.
	•	Prefetch fires at pause start + p10 − reload cost, via a warm-up request: the session's last prompt with max_tokens=1. LMCache serves the prefix and vLLM caches it in GPU blocks.
	•	The warm-up is cancelled if the agent has already returned; its blocks get a high score so they are not evicted before use.
	•	If LMCache's controller API can load KV for a prefix without a request, it replaces the warm-up.
C5. Trace recorder
	•	Wraps mini-SWE-agent's model and environment calls and writes one JSONL file per session.
	•	Per turn: full message list, prompt and output token counts, request start and end, tool command, tool start and end.
	•	Recording uses an API model through LiteLLM, on 50 to 100 SWE-bench Verified tasks, with Docker on the Mac and Apptainer on the cluster.
C6. Replay harness
	•	One asyncio coroutine per session, using httpx against vLLM's OpenAI-compatible API.
	•	Each turn sends the recorded message list, forces the recorded output length with min_tokens and max_tokens, then sleeps for the recorded tool duration.
	•	Sessions start by a Poisson process; the arrival rate is the load knob.
	•	Recorded assistant text (not the server's output) goes into the next prompt, so prompts are identical across configurations. The previous turn's generated tokens therefore miss the cache, a few hundred tokens per turn, equally for every policy.
C7. Simulator (Mac)
	•	Discrete-event loop on a heap of timestamped events; GPU and CPU block pools with fixed capacities.
	•	Prefill, decode, and transfer times from the C3 formulas; decode time per token from calibration.
	•	Runs any KVPolicy against the same traces as the harness, in seconds per run.
C8. Metrics
	•	Harness logs per request: send time, first-token time, last-token time, session, turn.
	•	Goodput = requests per second meeting the TTFT SLO (initial target: 1 s; tuned after calibration).
	•	From vLLM's Prometheus endpoint: KV cache usage and prefix-cache hit counters, from which re-prefilled tokens and GPU-seconds are derived.
Experiment plan
Five configurations replay the same traces at five load levels and two KV memory caps, three runs each; the noisy oracle runs only at the highest load and tight cap. That is about 160 runs of 12 minutes including warm-up and restart, roughly 32 GPU-hours.
Configuration
Implementation
Question it answers
vLLM default
Prefix caching, LRU, no LMCache
Status quo
Always offload
LMCache on, GPU blocks freed at every pause
Value of a CPU tier without prediction
Continuum
Their vLLM fork, default settings
Closest prior work: predicted TTL pinning
Ours, oracle
Policy fed true pause durations from the trace
Upper bound for any predictor
Ours, noisy oracle
True durations × (1 + error), error at ±25%, ±50%, ±100%
Sensitivity to prediction quality
Ours
Full system with the C2 predictor
The result
Load and memory knobs
	•	Load: Poisson session arrivals at five rates, chosen in calibration so the lowest rate meets the SLO under vLLM default and the highest violates it.
	•	Memory pressure: cap KV memory with --gpu-memory-utilization or a block-count override at two levels, one comfortable and one tight.
	•	Workload: 50 to 100 recorded sessions; 20% held out for predictor evaluation.
Procedure per run
	•	Start vLLM (and LMCache) with the configuration and memory cap; wait for the health check.
	•	Run a 2-minute warm-up replay; discard its metrics.
	•	Replay for 10 minutes at the target load; scrape Prometheus every 5 s.
	•	Stop, store logs under results/<config>/<load>/<cap>/<run>/.
	•	Restart the server between runs so no cache state carries over.
Outputs
	•	Goodput vs. load, one line per configuration, per memory cap.
	•	Returning-turn TTFT distribution (p50, p99) at the highest load.
	•	Re-prefilled GPU-seconds per configuration.
	•	Goodput vs. injected error for the noisy oracle.
	•	Predictor calibration: share of true pauses below p10 and below p90, per program.
	•	Simulator vs. measured goodput for each policy.
Implementation plan
GPU-free work runs on the Mac in weeks 1 and 2 so cluster time goes to vLLM integration and experiments from week 3. Dates assume a Dec 4 deadline; Thanksgiving (Nov 26) falls in the experiment week, so queue those runs as overnight jobs.

implementation timeline · 7 weeks, 3 gates
Exit criteria by week (where the work runs in parentheses)
	•	Week 1 (both): vLLM and LMCache pinned and serving Qwen2.5-Coder-7B on the cluster; warm-up prefetch tested; KVPolicy interface and predictor written with unit tests on the Mac.
	•	Week 2 (Mac): 50 to 100 sessions recorded as JSONL; simulator runs LRU, always-offload, and oracle policies; headroom gate decided.
	•	Week 3 (cluster): replay harness reproduces a session's prompts exactly; vLLM default and always-offload baselines measured; prefill rate and transfer bandwidth calibrated.
	•	Week 4 (cluster): session table and policy-driven eviction running in vLLM with the oracle; simulator compared against real runs.
	•	Week 5 (cluster): predictor and prefetch integrated; Continuum fork running on the same traces.
	•	Week 6 (cluster): all runs in the experiment plan complete, logs archived.
	•	Week 7 (Mac): figures generated by script from logs; report written in the LaTeX template.
Potential blockers
The three most likely blockers are version conflicts between vLLM, LMCache, and Continuum; cluster restrictions on containers; and SWE-bench contexts exceeding the model's window. Each has an early signal in weeks 1 to 2, so none should surface late.
Blocker
Likelihood
Early signal
Mitigation
Fallback
vLLM, LMCache, and the Continuum fork need incompatible vLLM versions
High
Install fails, or LMCache's connector errors at startup
Pin the vLLM version LMCache's docs list as supported; build ours on that
Drop the Continuum fork; reimplement predicted-TTL pinning as a KVPolicy
Cluster blocks Docker, and SWE-bench images fail under Apptainer
High
First Apptainer run of one SWE-bench task fails
Record traces on the Mac or a cloud VM with Docker; only replay needs the cluster
Use a non-container agent workload (e.g., repo tasks run in a plain venv)
GPU queue waits on the cluster stall experiments
Medium
Interactive GPU jobs wait over 1 hour in week 1
Batch runs into overnight Slurm jobs; develop policy in the simulator
Rent a cloud GPU for the final experiment week (about 32 GPU-hours)
Session contexts exceed Qwen2.5-Coder-7B's 32K native window
High
Trace token counts above 32K
Cap the recording agent's steps; drop or truncate sessions over 30K tokens
Enable the model's extended-context mode, at a cost in KV memory
No clean way to load KV into GPU before a request arrives
Medium
Warm-up request in week 1 shows no prefix-cache hit on the real turn
Raise the warm-up's eviction score; check LMCache's controller API
Evaluate placement and eviction without prefetch
Session ID does not reach the engine core process
Medium
ID missing in scheduler logs
Patch the API server to carry it in the request's sampling extra args
Encode the session ID in the request ID prefix
LMCache chunk size (256 tokens) and vLLM blocks (16 tokens) cause partial reloads
Medium
Hit rate lower than expected for returning turns
Align LMCache chunk size with the prompt growth per turn, or accept the tail recompute
Report it as overhead common to all LMCache configurations
Oracle policy barely beats always-offload
Medium
Simulator gap under 5% goodput in week 2
Test tighter memory caps, longer contexts, disk tier
Reframe as a characterization of when prediction helps
Run-to-run variance hides differences between policies
Medium
Pilot runs differ by over 10% goodput
Fixed seeds, 3 runs, restart between runs, report 95% intervals
More runs at fewer load levels
Recording API cost too high
Low
First 10 tasks cost over 15% of budget
Cheaper model, fewer tasks, provider prompt caching
Use published trajectories and synthesize tool durations from per-command distributions
Recorded tool durations distorted by x86 emulation on Apple Silicon
Medium
Mac pytest runs much slower than the same task on the cluster
Record the final traces on x86 hardware
Scale durations by a measured correction factor
Coursework and job hours compress the schedule
Medium
Week 3 milestone missed
Apply the scope cuts in the next section in order
Submit placement plus oracle results only
Scope cuts and decision points
Four checkpoints decide whether the plan continues as written; if the schedule slips, cuts apply in a fixed order.
Decision points
When
Check
If it passes
If it fails
End of week 1
Pinned versions of vLLM and LMCache run together on the cluster GPU
Build on that version
Drop Continuum's fork; pin to LMCache's supported vLLM
End of week 1
Warm-up request produces a prefix-cache hit on the following real request
Keep prefetch in scope
Cut prefetch (cut 3 below)
End of week 2
Simulated oracle beats always-offload by at least 5% goodput at the tight cap
Proceed with the full system
Reframe as a characterization study; keep the same experiments
End of week 4
Simulator goodput within 15% of real vLLM for the oracle policy
Use the simulator for tuning
Tune on real runs only; report the gap
Scope cuts, in order
	•	Disk tier (already out of scope unless ahead of schedule).
	•	Noisy-oracle sweep reduced to one error level (±50%).
	•	Predictive prefetch; evaluate placement and eviction alone.
	•	Continuum baseline replaced by the reimplemented predicted-TTL policy.
	•	Load sweep reduced from five levels to three.
After all five cuts the project still delivers its core claim: oracle and predicted placement against vLLM default and always-offload.
