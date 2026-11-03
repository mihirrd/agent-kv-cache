"""Pick experiment cells from earlier replay results, so a queue on a rented GPU never waits for a person.

- `load`: the highest probed load the server keeps up with (full KV pool).
- `pools`: KV pool sizes as fractions of the time-averaged live KV (running requests plus paused
  sessions) in a full-pool run at that load. On the 0.5B emulator at load 8 (live KV 27-36K tokens)
  the pools where eviction happened without collapse, 45K and 60K, were about 1.5x and 2x live KV;
  the pool also holds reply blocks no prompt reuses and absorbs bursts above the average.

    uv run python -m brb.plan load results/replay/a100-probe 4 8 12 16
    uv run python -m brb.plan pools results/replay/a100-probe/load8/full/seed0 1.5 2
"""

import argparse
import json
import sys
from pathlib import Path

# A load "keeps up" when returning turns meet the 2 s target, requests barely queue, and turns are not
# stretching: an overloaded server serves every turn slower, so agents spend less of their time in tool
# calls (7B on the 2080 Ti: pause share 30% at load 2, 15% at load 4, where it fell behind). A short run
# can miss the queue building up but not the stretched turns.
MIN_SLO_SHARE_2S = 0.98
MAX_WAITING_AVG = 1.0
MIN_PAUSE_SHARE_RATIO = 0.7  # vs the lowest probed load


def keeps_up(summary: dict, base_pause_share: float | None = None) -> bool:
    pause = summary.get("pause_share") or 0
    return (summary.get("errors", 0) == 0 and (summary.get("returning_slo_share_2s") or 0) >= MIN_SLO_SHARE_2S
            and (summary.get("waiting_avg") or 0) <= MAX_WAITING_AVG
            and (base_pause_share is None or pause >= MIN_PAUSE_SHARE_RATIO * base_pause_share))


def pick_load(summaries: dict[float, dict]) -> float | None:
    """Highest load that keeps up, among loads below the first one that does not."""
    best, base = None, None
    for load in sorted(summaries):
        if not keeps_up(summaries[load], base):
            break
        best = load
        if base is None:
            base = summaries[load].get("pause_share")
    return best


def live_kv(records: list[dict], t0: float, t1: float, step: float = 1.0) -> tuple[float, float]:
    """Time-averaged KV tokens of running requests and of paused sessions over [t0, t1). A request holds
    its prompt plus reply while it runs; a paused session holds the same until its next request.
    A session's last request ends it."""
    by_sid: dict[str, list[dict]] = {}
    for r in records:
        by_sid.setdefault(r["sid"], []).append(r)
    running_iv, paused_iv = [], []  # (start, end, tokens)
    for rs in by_sid.values():
        rs.sort(key=lambda r: r["turn"])
        for i, r in enumerate(rs):
            if r.get("finish") is None:
                running_iv.append((r["arrival"], t1, r["prompt_tokens"]))
                continue
            tokens = r["prompt_tokens"] + r["output_tokens"]
            running_iv.append((r["arrival"], r["finish"], tokens))
            if i + 1 < len(rs):
                paused_iv.append((r["finish"], rs[i + 1]["arrival"], tokens))
    n = max(1, int((t1 - t0) / step))
    ts = [t0 + (k + 0.5) * step for k in range(n)]

    def avg(ivs):
        return sum(tok for t in ts for a, b, tok in ivs if a <= t < b) / n

    return avg(running_iv), avg(paused_iv)


def pools(run_dir: Path, fractions: list[float]) -> tuple[list[int], float, float]:
    cfg = json.loads((run_dir / "config.json").read_text())
    records = [json.loads(line) for line in (run_dir / "requests.jsonl").open()]
    running, paused = live_kv(records, cfg["warmup_s"], cfg["warmup_s"] + cfg["duration_s"])
    return [round(f * (running + paused) / 1000) * 1000 for f in fractions], running, paused


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m brb.plan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("load", help="print the highest load that keeps up (nothing if none)")
    a.add_argument("config_dir", type=Path, help="results/replay/<config>, with load<L>/full/seed0 runs")
    a.add_argument("loads", type=float, nargs="+")
    b = sub.add_parser("pools", help="print pool sizes in tokens, one per fraction of live KV")
    b.add_argument("run_dir", type=Path)
    b.add_argument("fractions", type=float, nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "load":
        summaries = {}
        for load in args.loads:
            path = args.config_dir / f"load{load:g}" / "full" / "seed0" / "summary.json"
            if path.exists():
                summaries[load] = json.loads(path.read_text())
        best = pick_load(summaries)
        if best is not None:
            print(f"{best:g}")
    else:
        sizes, running, paused = pools(args.run_dir, args.fractions)
        print(f"live KV: running {running:,.0f} + paused {paused:,.0f} tokens", file=sys.stderr)
        print(" ".join(str(s) for s in sizes))


if __name__ == "__main__":
    main()
