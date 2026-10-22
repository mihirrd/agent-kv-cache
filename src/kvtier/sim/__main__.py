"""C7 simulator CLI (run on gpubox).

  uv run python -m kvtier.sim run --policy oracle --load 20 --cap 100000 --seed 1
  uv run python -m kvtier.sim sweep --policy vllm always-offload oracle --load 10 20 40 --cap 100000 205936
"""

import argparse
import concurrent.futures
import dataclasses
import json
import random
import subprocess
import time
from pathlib import Path

from kvtier.cost import GATE_A_TP2, Calibration, CostModel, PredictivePolicy
from kvtier.sessions import common_prefix_blocks, load_sessions
from kvtier.sim.engine import SimConfig, Simulator
from kvtier.sim.metrics import summarize, write_outputs
from kvtier.sim.policies import (EvictOnlyOracle, EvictOnlyPredictive, LongCallOffloadPolicy, LruPolicy, OraclePolicy,
                                 RemainingTimeEviction, TtlPinPolicy, warm_start)

POLICIES = ("vllm", "always-offload", "ttl-pin", "predictive", "predictive-cold", "oracle", "noisy-oracle",
            "oracle-evict", "predictive-evict", "long-offload", "long-offload-oracle", "remaining-evict", "belady")
DEFAULT_SESSIONS = ("traces/sessions/verified.jsonl", "traces/sessions/multilingual.jsonl")


def split(sessions_by_file: list[list], which: str, train_share: float = 0.8, seed: int = 1234) -> list:
    """Fixed 80/20 split, stratified by file (trace set)."""
    out = []
    for sessions in sessions_by_file:
        ordered = sorted(sessions, key=lambda s: s.session_id)
        random.Random(seed).shuffle(ordered)
        k = round(train_share * len(ordered))
        out += {"train": ordered[:k], "holdout": ordered[k:], "all": ordered}[which]
    return out


def faster_compute(cal: Calibration, speedup: float) -> Calibration:
    """A GPU `speedup` times faster at prefill and decode, with the same KV memory and CPU-reload cost."""
    if speedup == 1:
        return cal
    return dataclasses.replace(
        cal,
        prefill_points=tuple((t, s / speedup) for t, s in cal.prefill_points),
        decode_base_s=cal.decode_base_s / speedup,
        decode_per_request_s=cal.decode_per_request_s / speedup,
    )


def make_policy(name: str, cost: CostModel, train, ttl_s: float, noise: float, seed: int, threshold_s: float = 5.0):
    if name in ("vllm", "always-offload"):
        return LruPolicy()
    if name == "ttl-pin":
        return TtlPinPolicy(ttl_s)
    if name in ("predictive", "predictive-cold"):
        p = PredictivePolicy(cost)
        if name == "predictive":
            warm_start(p, train)
        return p
    if name == "oracle":
        return OraclePolicy(cost)
    if name == "noisy-oracle":
        return OraclePolicy(cost, noise=noise, seed=seed)
    if name == "long-offload":
        p = LongCallOffloadPolicy(threshold_s)
        warm_start(p, train)
        return p
    if name == "long-offload-oracle":
        return LongCallOffloadPolicy(threshold_s, oracle=True)
    if name == "remaining-evict":
        p = RemainingTimeEviction()
        warm_start(p, train)
        return p
    if name == "belady":
        return RemainingTimeEviction(oracle=True)
    if name == "oracle-evict":
        return EvictOnlyOracle(cost)
    if name == "predictive-evict":
        p = EvictOnlyPredictive(cost)
        warm_start(p, train)
        return p
    raise ValueError(f"unknown policy {name!r}; choose from {POLICIES}")


def run_one(policy: str, load: float, cap: int, seed: int, opts: dict) -> dict:
    by_file = [load_sessions(Path(p))[0] for p in opts["sessions"]]
    sessions = split(by_file, opts["split"])
    train = split(by_file, "train")
    cost = CostModel(faster_compute(GATE_A_TP2, opts["speedup"]))
    cfg = SimConfig(load_per_min=load, cap_tokens=cap, cpu_tier=policy != "vllm", prefetch=not opts["no_prefetch"],
                    warmup_s=opts["warmup"], duration_s=opts["duration"], seed=seed)
    pol = make_policy(policy, cost, train, opts["ttl"], opts["noise"], seed, opts["threshold"])
    t0 = time.time()
    rec = Simulator(sessions, pol, cost, cfg, common_prefix_blocks(sessions)).run()
    summary = {"policy": policy, "load_per_min": load, "cap_tokens": cap, "seed": seed,
               **summarize(rec, cfg), "wall_s": round(time.time() - t0, 2)}
    out = Path(opts["out"]) / policy / f"load{load:g}" / f"cap{cap}" / f"seed{seed}"
    config = {**dataclasses.asdict(cfg), "policy": policy, "calibration": "GATE_A_TP2", **opts, "git": _git()}
    write_outputs(out, rec, summary, config)
    return summary


def _git() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("run", "sweep"))
    ap.add_argument("--policy", nargs="+", default=["oracle"], choices=POLICIES)
    ap.add_argument("--load", nargs="+", type=float, default=[20.0], help="session arrivals per minute")
    ap.add_argument("--cap", nargs="+", type=int, default=[205_936], help="GPU KV cap in tokens")
    ap.add_argument("--seed", nargs="+", type=int, default=[0])
    ap.add_argument("--sessions", nargs="+", default=list(DEFAULT_SESSIONS))
    ap.add_argument("--split", choices=("train", "holdout", "all"), default="train")
    ap.add_argument("--ttl", type=float, default=5.0, help="TTL-pin seconds")
    ap.add_argument("--noise", type=float, default=0.5, help="noisy-oracle relative error")
    ap.add_argument("--threshold", type=float, default=5.0, help="long-offload: pauses longer than this are long")
    ap.add_argument("--speedup", type=float, default=1.0, help="prefill and decode this many times faster (same memory)")
    ap.add_argument("--no-prefetch", action="store_true")
    ap.add_argument("--warmup", type=float, default=120.0)
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--out", default="results/sim")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    opts = {k: getattr(args, k) for k in ("sessions", "split", "ttl", "noise", "threshold", "speedup", "no_prefetch",
                                          "warmup", "duration", "out")}
    grid = [(p, l, c, s) for p in args.policy for l in args.load for c in args.cap for s in args.seed]
    if args.mode == "run" and len(grid) > 1:
        ap.error("run takes one value per option; use sweep for a grid")
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, len(grid))) as pool:
        results = list(pool.map(run_one, *zip(*grid), [opts] * len(grid)))
    keys = ("policy", "load_per_min", "cap_tokens", "seed", "goodput", "goodput_5s", "returning_ttft_p50_s",
            "returning_ttft_p99_s", "returning_recompute_share", "re_prefill_gpu_s", "warmups_issued", "wall_s")
    print("\t".join(keys))
    for r in sorted(results, key=lambda r: (r["cap_tokens"], r["load_per_min"], r["policy"], r["seed"])):
        print("\t".join(f"{r[k]:.3f}" if isinstance(r[k], float) else str(r[k]) for k in keys))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with (Path(args.out) / f"{args.mode}_{int(time.time())}.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
