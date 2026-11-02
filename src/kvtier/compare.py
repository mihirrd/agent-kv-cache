"""Compare eviction policies across pools and seeds: returning-turn TTFT, recompute and gap closed.

Usage: python -m kvtier.compare <name> [policies...]
Reads results/replay/<name>-<policy>/load*/<pool>/seed*/ (requests.jsonl, config.json). Each seed's
metrics use the measurement window only; the table shows the mean over seeds, and the pooled
columns put every seed's returning turns in one distribution.
"""
import json
import statistics
import sys
from pathlib import Path

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


if __name__ == "__main__":
    main()
