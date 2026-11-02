"""Compare eviction policies across pools and seeds: returning-turn TTFT, recompute and gap closed,
then job completion time (Continuum's metric).

Usage: python -m kvtier.compare <name> [policies...]
Reads results/replay/<name>-<policy>/load*/<pool>/seed*/ (requests.jsonl, config.json). Each seed's
metrics use the measurement window only; the table shows the mean over seeds, and the pooled
columns put every seed's returning turns in one distribution. Job times cover sessions that arrived
in the window and finished every turn under every policy (same seed, same sessions), so each
policy is timed on the same jobs.
"""
import json
import statistics
import sys
from functools import cache
from pathlib import Path

from kvtier.sessions import load_sessions

ROOT = Path("results/replay")
QUANTILES = (0.5, 0.9, 0.99)


def _q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def returning(run: Path) -> list[dict]:
    cfg = json.loads((run / "config.json").read_text())
    lo, hi = cfg["warmup_s"], cfg["warmup_s"] + cfg["duration_s"]
    recs = [json.loads(line) for line in (run / "requests.jsonl").open()]
    return [r for r in recs if lo <= r["arrival"] < hi and r["turn"] > 0 and r["ttft"] is not None]


def metrics(ret: list[dict]) -> dict:
    ttft = [r["ttft"] for r in ret]
    m = {f"p{round(p * 100)}": _q(ttft, p) for p in QUANTILES}
    m["le1s"] = sum(t <= 1 for t in ttft) / len(ttft)
    # prompt tokens the GPU had to recompute: what eviction costs
    m["recompute_k"] = sum(r["served_prompt_tokens"] - r["cached_tokens"] for r in ret) / 1000
    m["n"] = len(ret)
    return m


@cache
def turn_counts(session_files: tuple[str, ...]) -> dict[str, int]:
    return {s.session_id: len(s.turns) for f in session_files for s in load_sessions(Path(f))[0]}


def jobs(run: Path) -> dict[str, tuple[float, float]]:
    """sid -> (job completion time, time spent in the model) for sessions that arrived in the window
    and finished every turn."""
    cfg = json.loads((run / "config.json").read_text())
    n_turns = turn_counts(tuple(cfg["sessions"]))
    lo, hi = cfg["warmup_s"], cfg["warmup_s"] + cfg["duration_s"]
    by_sid: dict[str, list[dict]] = {}
    for line in (run / "requests.jsonl").open():
        r = json.loads(line)
        by_sid.setdefault(r["sid"], []).append(r)
    out = {}
    for sid, recs in by_sid.items():
        limit = cfg.get("max_turns") or n_turns[recs[0]["trace"]]
        done = [r for r in recs if r.get("finish") is not None]
        if lo <= recs[0]["session_arrival"] < hi and len(done) == min(limit, n_turns[recs[0]["trace"]]):
            out[sid] = (max(r["finish"] for r in done) - recs[0]["session_arrival"],
                        sum(r["finish"] - r["arrival"] for r in done))
    return out


def job_table(name: str, policies: list[str]) -> None:
    runs: dict[tuple[str, str], dict[str, dict]] = {}  # (pool, seed) -> policy -> jobs
    for policy in policies:
        for seed_dir in ROOT.glob(f"{name}-{policy}/load*/*/seed*"):
            runs.setdefault((seed_dir.parent.name, seed_dir.name), {})[policy] = jobs(seed_dir)
    print(f"\n{'pool':>7} {'policy':>9} {'jobs':>5} {'JCT mean':>9} {'JCT p90':>8} {'model s/job':>11}")
    for pool in sorted({p for p, _ in runs}, key=int):
        cells = [by_policy for (p, _), by_policy in runs.items() if p == pool and len(by_policy) == len(policies)]
        common = [set.intersection(*(set(c[pol]) for pol in policies)) for c in cells]
        for policy in policies:
            jct = [c[policy][sid][0] for c, sids in zip(cells, common) for sid in sids]
            model = [c[policy][sid][1] for c, sids in zip(cells, common) for sid in sids]
            if jct:
                print(f"{pool:>7} {policy:>9} {len(jct):>5} {statistics.mean(jct):9.1f} {_q(jct, 0.9):8.1f} "
                      f"{statistics.mean(model):11.2f}")


def main() -> None:
    name, policies = sys.argv[1], sys.argv[2:] or ["lru", "remaining", "oracle"]
    table: dict[tuple[str, str], dict] = {}
    for policy in policies:
        for pool_dir in sorted(ROOT.glob(f"{name}-{policy}/load*/*")):
            seeds = sorted(pool_dir.glob("seed*"))
            per_seed = [metrics(returning(s)) for s in seeds]
            row = {k: statistics.mean(m[k] for m in per_seed) for k in per_seed[0]}
            row["p99_sd"] = statistics.stdev(m["p99"] for m in per_seed) if len(per_seed) > 1 else 0.0
            pooled = metrics([r for s in seeds for r in returning(s)])
            row["pooled_p99"] = pooled["p99"]
            row["seeds"] = len(seeds)
            table[(pool_dir.name, policy)] = row

    pools = sorted({pool for pool, _ in table}, key=int)
    print(f"{'pool':>7} {'policy':>9} {'seeds':>5} {'p50':>6} {'p90':>6} {'p99':>6} {'±sd':>5} "
          f"{'pooled99':>8} {'≤1s':>7} {'recomp K':>9}")
    for pool in pools:
        for policy in policies:
            r = table.get((pool, policy))
            if r:
                print(f"{pool:>7} {policy:>9} {r['seeds']:>5} {r['p50']:6.3f} {r['p90']:6.3f} {r['p99']:6.3f} "
                      f"{r['p99_sd']:5.2f} {r['pooled_p99']:8.3f} {r['le1s']:7.2%} {r['recompute_k']:9.0f}")
        if {"lru", "remaining", "oracle"} <= set(policies):
            lru, ours, orc = (table[(pool, p)] for p in ("lru", "remaining", "oracle"))
            for key in ("pooled_p99", "recompute_k"):
                span = lru[key] - orc[key]
                gap = (lru[key] - ours[key]) / span if span else float("nan")
                print(f"{'':>7} gap closed on {key}: {gap:.0%}")
    job_table(name, policies)


if __name__ == "__main__":
    main()
