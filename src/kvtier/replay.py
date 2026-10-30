"""C6 replay harness: replay recorded agent sessions against a live vLLM server and log time to first token.

Sessions arrive as a Poisson process drawn exactly as the simulator draws them, so for the same seed
a real run and a simulated run see the same sessions at the same times (Gate D). Each turn sends the
recorded messages up to that turn, forces the recorded output length (min_tokens = max_tokens), then
sleeps for the recorded tool time. Every session copy gets a salt line at the start of its task
message: copies share only the system prompt, as in the simulator, and nothing carries over between runs.

On gpubox, with the server up:
  uv run python -m kvtier.replay check                  # replayed prompts match the trace token for token
  uv run python -m kvtier.replay run --load 6 --seed 0  # writes results/replay/<config>/load6/<cap>/seed0/
"""

import argparse
import asyncio
import dataclasses
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from kvtier.predictor import percentile
from kvtier.sessions import ReplaySession, block_hashes, load_sessions, split

DEFAULT_SESSIONS = ("traces/sessions/verified.jsonl", "traces/sessions/multilingual.jsonl")
SLOS_S = (2.0, 5.0, 10.0)  # same as kvtier.sim.metrics
HEADLINE_SLO_S = 5.0
# Prometheus series scraped during a run (C8); values are summed over label sets.
METRICS = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc", "num_preemptions_total",
           "prefix_cache_queries_total", "prefix_cache_hits_total",
           "external_prefix_cache_queries_total", "external_prefix_cache_hits_total")


@dataclass
class ReplayTrace:
    session: ReplaySession
    messages: list[dict]  # the whole recorded conversation
    prompt_messages: list[int]  # per turn: how many leading messages form its prompt

    @property
    def session_id(self) -> str:
        return self.session.session_id

    def prompt(self, turn: int, salt: str = "") -> list[dict]:
        msgs = self.messages[: self.prompt_messages[turn]]
        if not salt:
            return msgs
        i = next(k for k, m in enumerate(msgs) if m["role"] == "user")
        return msgs[:i] + [{**msgs[i], "content": salt + msgs[i]["content"]}] + msgs[i + 1 :]


def attach_messages(session: ReplaySession, trace_path: Path) -> ReplayTrace:
    """Pair a replay session with the recorded messages of its raw trace."""
    lines = [json.loads(line) for line in Path(trace_path).open()]
    turns = [r for r in lines if r["type"] == "turn"][: len(session.turns)]
    messages = lines[-1]["messages"] if lines[-1]["type"] == "footer" else []
    return ReplayTrace(session, messages, [r["prompt_messages"] for r in turns])


def load_traces(session_files, which: str) -> list[ReplayTrace]:
    """Replay sessions with their messages, split and ordered exactly as the simulator orders them."""
    by_file = []
    for p in session_files:
        sessions, meta = load_sessions(Path(p))
        source = Path(meta["source"])
        by_file.append([attach_messages(s, source / f"{s.session_id}.jsonl") for s in sessions])
    return split(by_file, which)


def arrival_schedule(n_sessions: int, load_per_min: float, horizon_s: float, seed: int) -> list[tuple[float, int, int]]:
    """(time, session index, salt) of each arrival up to horizon_s, in the simulator's order of draws."""
    rng = random.Random(seed)
    rate = load_per_min / 60
    out = []
    t = rng.expovariate(rate)
    while t <= horizon_s:
        i = rng.choice(range(n_sessions))
        salt = rng.getrandbits(62)
        out.append((t, i, salt))
        t += rng.expovariate(rate)
    return out


def parse_metrics(text: str) -> dict[str, float]:
    out = {}
    for name in METRICS:
        values = re.findall(rf"^vllm:{name}(?:\{{.*\}})? (\S+)$", text, re.M)
        if values:
            out[name] = sum(float(v) for v in values)
    return out


@dataclass
class RunConfig:
    load_per_min: float = 6.0  # session arrivals per minute
    warmup_s: float = 120.0
    duration_s: float = 600.0
    drain_s: float = 120.0  # keep running after the window so in-window requests can finish
    seed: int = 0
    priority: str = "none"  # "session-arrival": priority = ms from run start to the session's arrival
    pause_scale: float = 1.0  # multiply recorded tool pauses (tests use 0)
    # Stretch the clock k-fold: pauses and arrival gaps take k times longer on the wall clock and every
    # measured time is divided by k, so the GPU behaves as if it were k times faster (notes/emulation.md).
    time_scale: float = 1.0
    max_turns: int | None = None
    scrape_every_s: float = 5.0
    base_url: str = "http://localhost:8000"


class Replayer:
    def __init__(self, traces: list[ReplayTrace], cfg: RunConfig, client: httpx.AsyncClient, model: str):
        self.traces, self.cfg, self.client, self.model = traces, cfg, client, model
        self.records: list[dict] = []  # one per request, added when it is sent
        self.metrics: list[dict] = []
        self.t0 = 0.0

    def now(self) -> float:
        """Time since the start of the run, in emulated seconds (wall seconds / time_scale)."""
        return (time.monotonic() - self.t0) / self.cfg.time_scale

    async def sleep(self, emulated_s: float) -> None:
        await asyncio.sleep(max(0.0, emulated_s) * self.cfg.time_scale)

    async def run(self) -> list[dict]:
        cfg = self.cfg
        end = cfg.warmup_s + cfg.duration_s + cfg.drain_s
        schedule = arrival_schedule(len(self.traces), cfg.load_per_min, end, cfg.seed)
        self.t0 = time.monotonic()
        scraper = asyncio.create_task(self.scrape())
        tasks = []
        for k, (t, i, salt) in enumerate(schedule):
            await self.sleep(t - self.now())
            tr = self.traces[i]
            tasks.append(asyncio.create_task(self.session(tr, f"{tr.session_id}#{k}", salt, t)))
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=max(0.0, end - self.now()) * cfg.time_scale)
            for task in pending:
                task.cancel()  # closing the stream aborts the request in vLLM
            await asyncio.gather(*pending, return_exceptions=True)
        scraper.cancel()
        await asyncio.gather(scraper, return_exceptions=True)
        return self.records

    async def session(self, tr: ReplayTrace, sid: str, salt: int, arrived: float) -> None:
        prio = int(arrived * 1000) if self.cfg.priority == "session-arrival" else None
        tag = f"(replay {salt:016x})\n"
        turns = tr.session.turns[: self.cfg.max_turns]
        pause_before = None
        for turn, st in enumerate(turns):
            rec = {"sid": sid, "trace": tr.session_id, "turn": turn, "session_arrival": arrived,
                   "arrival": self.now(), "first_token": None, "finish": None, "ttft": None,
                   "prompt_tokens": st.prompt_tokens, "output_tokens": max(1, st.output_tokens),
                   "program": st.program, "pause_before_s": pause_before, "priority": prio}
            self.records.append(rec)
            await self.request(tr.prompt(turn, tag), sid, prio, rec)
            if "error" in rec:
                return
            if st.pause_s:
                pause_before = st.pause_s * self.cfg.pause_scale
                await self.sleep(pause_before)

    async def request(self, messages: list[dict], sid: str, prio: int | None, rec: dict) -> None:
        n = rec["output_tokens"]
        body = {"model": self.model, "messages": messages, "max_tokens": n, "min_tokens": n, "temperature": 0,
                "stream": True, "stream_options": {"include_usage": True}, "vllm_xargs": {"session_id": sid}}
        if prio is not None:
            body["priority"] = prio
        try:
            async with self.client.stream("POST", f"{self.cfg.base_url}/v1/chat/completions", json=body) as r:
                if r.status_code != 200:
                    rec["error"] = f"HTTP {r.status_code}: {(await r.aread())[:300]!r}"
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[len("data: "):])
                    # vLLM sends the first chunk once the first output token exists (after prefill).
                    if chunk.get("choices") and rec["first_token"] is None:
                        rec["first_token"] = self.now()
                        rec["ttft"] = rec["first_token"] - rec["arrival"]
                    if usage := chunk.get("usage"):
                        rec["served_prompt_tokens"] = usage["prompt_tokens"]
                        rec["served_output_tokens"] = usage["completion_tokens"]
                        rec["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            rec["finish"] = self.now()
        except httpx.HTTPError as e:
            rec["error"] = repr(e)

    async def scrape(self) -> None:
        while True:
            try:
                r = await self.client.get(f"{self.cfg.base_url}/metrics")
                self.metrics.append({"t": self.now(), **parse_metrics(r.text)})
            except httpx.HTTPError:
                pass
            await asyncio.sleep(self.cfg.scrape_every_s)


def _q(xs, f):
    return percentile(sorted(xs), f) if xs else None


def summarize(records: list[dict], cfg: RunConfig) -> dict:
    """Same headline metrics as the simulator; requests sent in the window but never served count as misses."""
    lo, hi = cfg.warmup_s, cfg.warmup_s + cfg.duration_s
    window = [r for r in records if lo <= r["arrival"] < hi]
    served = [r for r in window if r["ttft"] is not None]
    ret = [r for r in served if r["turn"] > 0]
    n_ret = sum(r["turn"] > 0 for r in window)
    s = {
        "requests": len(window), "returning_requests": n_ret, "served": len(served),
        "errors": sum("error" in r for r in records),
        "ttft_p50_s": _q([r["ttft"] for r in served], 0.5), "ttft_p99_s": _q([r["ttft"] for r in served], 0.99),
        "returning_ttft_p50_s": _q([r["ttft"] for r in ret], 0.5),
        "returning_ttft_p99_s": _q([r["ttft"] for r in ret], 0.99),
        # new sessions' first-token wait: the cost of a queue-order change
        "first_turn_ttft_p50_s": _q([r["ttft"] for r in served if r["turn"] == 0], 0.5),
        "first_turn_ttft_p99_s": _q([r["ttft"] for r in served if r["turn"] == 0], 0.99),
    }
    for slo in SLOS_S:
        s[f"goodput_{slo:g}s"] = sum(r["ttft"] <= slo for r in served) / cfg.duration_s
        s[f"returning_goodput_{slo:g}s"] = sum(r["ttft"] <= slo for r in ret) / cfg.duration_s
        s[f"returning_slo_share_{slo:g}s"] = sum(r["ttft"] <= slo for r in ret) / n_ret if n_ret else None
    s["goodput"] = s[f"returning_goodput_{HEADLINE_SLO_S:g}s"]  # headline: returning turns at 5 s
    # what an agent waits for per turn: first token plus writing the reply
    done = [r["finish"] - r["arrival"] for r in ret if r.get("finish") is not None]
    s["returning_turn_time_p50_s"] = _q(done, 0.5)
    s["returning_turn_time_p99_s"] = _q(done, 0.99)

    # Regime (notes/emulation.md): how much of an agent's life is tool time, and how much of returning
    # turns' prompts the GPU still held.
    paused = sum(r.get("pause_before_s") or 0 for r in window)
    busy = sum(r["finish"] - r["arrival"] for r in window if r.get("finish") is not None)
    s["pause_share"] = paused / (paused + busy) if paused + busy else None
    cached = [r for r in ret if r.get("cached_tokens") is not None]
    s["returning_cached_share"] = (sum(r["cached_tokens"] for r in cached) / sum(r["served_prompt_tokens"] for r in cached)
                                   if cached else None)

    # Sanity checks: the prompt differs from the trace only by the salt line; output length is forced.
    deltas = [r["served_prompt_tokens"] - r["prompt_tokens"] for r in records if "served_prompt_tokens" in r]
    s["prompt_token_delta_min"] = min(deltas, default=None)
    s["prompt_token_delta_max"] = max(deltas, default=None)
    s["output_length_mismatches"] = sum(
        r["served_output_tokens"] != r["output_tokens"] for r in records if "served_output_tokens" in r)
    return s


def regime(metrics: list[dict], cfg: RunConfig) -> dict:
    """Server-side view of the measurement window: is compute or memory the bottleneck?"""
    lo, hi = cfg.warmup_s, cfg.warmup_s + cfg.duration_s
    ms = [m for m in metrics if lo <= m["t"] < hi]
    if len(ms) < 3:
        return {}
    third = max(1, len(ms) // 3)
    avg = lambda key, xs=ms: sum(m.get(key, 0.0) for m in xs) / len(xs)  # noqa: E731
    delta = lambda key: ms[-1].get(key, 0.0) - ms[0].get(key, 0.0)  # noqa: E731
    queries = delta("prefix_cache_queries_total")
    return {
        "running_avg": avg("num_requests_running"),
        # > ~1.3: requests pile up over the window, i.e. the server cannot keep up (no steady state)
        "running_growth": avg("num_requests_running", ms[-third:]) / max(avg("num_requests_running", ms[:third]), 1.0),
        "waiting_avg": avg("num_requests_waiting"),
        "kv_usage_avg": avg("kv_cache_usage_perc"),
        "kv_usage_max": max(m.get("kv_cache_usage_perc", 0.0) for m in ms),
        "gpu_prefix_hit_rate": delta("prefix_cache_hits_total") / queries if queries else None,
        "cpu_reloaded_tokens": delta("external_prefix_cache_hits_total"),
        "preemptions": delta("num_preemptions_total"),
    }


async def _run(traces: list[ReplayTrace], cfg: RunConfig) -> tuple[list[dict], list[dict]]:
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=httpx.Timeout(None), limits=limits) as client:
        model = (await client.get(f"{cfg.base_url}/v1/models")).json()["data"][0]["id"]
        rp = Replayer(traces, cfg, client, model)
        await rp.run()
        return rp.records, rp.metrics


def check(traces: list[ReplayTrace], base_url: str) -> tuple[int, list[str]]:
    """Tokenize every replayed prompt on the server; compare with the trace's token count and block hashes."""
    client = httpx.Client(timeout=300)
    model = client.get(f"{base_url}/v1/models").json()["data"][0]["id"]
    checked, problems = 0, []
    for tr in traces:
        for turn, st in enumerate(tr.session.turns):
            r = client.post(f"{base_url}/tokenize",
                            json={"model": model, "messages": tr.prompt(turn), "add_generation_prompt": True})
            r.raise_for_status()
            tokens = r.json()["tokens"]
            checked += 1
            if len(tokens) != st.prompt_tokens or block_hashes(tokens) != st.block_hashes:
                problems.append(f"{tr.session_id} turn {turn}: {len(tokens)} tokens, trace has {st.prompt_tokens}")
    return checked, problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("run", "check"))
    ap.add_argument("--sessions", nargs="+", default=list(DEFAULT_SESSIONS))
    ap.add_argument("--split", choices=("train", "holdout", "all"), default=None,
                    help="default: train for run (as the simulator), all for check")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--load", type=float, default=6.0, help="session arrivals per minute")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=float, default=120.0)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--drain", type=float, default=120.0)
    ap.add_argument("--priority", choices=("none", "session-arrival"), default="none",
                    help="session-arrival needs the server started with --scheduling-policy priority")
    ap.add_argument("--pause-scale", type=float, default=1.0)
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="emulate a k-times faster GPU: wall clock runs k times longer, times are reported / k")
    ap.add_argument("--max-turns", type=int, default=None)
    ap.add_argument("--config", default="always-offload", help="label for the server setup, used in --out")
    ap.add_argument("--cap", default="full", help="label for the KV cap, used in --out")
    ap.add_argument("--out", default=None, help="default: results/replay/<config>/load<load>/<cap>/seed<seed>")
    args = ap.parse_args()

    if args.mode == "check":
        traces = load_traces(args.sessions, args.split or "all")
        checked, problems = check(traces, args.base_url)
        print("\n".join(problems))
        print(f"{checked} prompts checked, {len(problems)} mismatches: {'PASS' if not problems else 'FAIL'}")
        raise SystemExit(1 if problems else 0)

    traces = load_traces(args.sessions, args.split or "train")
    cfg = RunConfig(load_per_min=args.load, warmup_s=args.warmup, duration_s=args.duration, drain_s=args.drain,
                    seed=args.seed, priority=args.priority, pause_scale=args.pause_scale,
                    time_scale=args.time_scale, max_turns=args.max_turns, base_url=args.base_url)
    out = Path(args.out or f"results/replay/{args.config}/load{args.load:g}/{args.cap}/seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records, metrics = asyncio.run(_run(traces, cfg))
    summary = {"config": args.config, "cap": args.cap, "load_per_min": args.load, "seed": args.seed,
               "time_scale": args.time_scale, **summarize(records, cfg), **regime(metrics, cfg),
               "wall_s": round(time.time() - started, 1)}
    for name, rows in (("requests.jsonl", records), ("metrics.jsonl", metrics)):
        (out / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "config.json").write_text(json.dumps({**dataclasses.asdict(cfg), "sessions": args.sessions,
                                                 "split": args.split or "train", "config": args.config,
                                                 "cap": args.cap, "started": started}, indent=2))
    for k in ("requests", "served", "errors", "goodput", "goodput_5s", "returning_ttft_p50_s", "returning_ttft_p99_s",
              "returning_turn_time_p50_s", "first_turn_ttft_p50_s", "pause_share", "returning_cached_share",
              "running_growth", "kv_usage_max", "cpu_reloaded_tokens", "preemptions", "output_length_mismatches"):
        print(f"{k:<26} {summary.get(k)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
