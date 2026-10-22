"""Summary metrics and output files for one simulator run (C7 spec section 8)."""

import csv
import json
from pathlib import Path

from kvtier.predictor import percentile
from kvtier.sessions import BLOCK
from kvtier.sim.engine import Records, SimConfig

SLOS_S = (2.0, 5.0, 10.0)
HEADLINE_SLO_S = 5.0


def _q(xs, f):
    return percentile(sorted(xs), f) if xs else None


def summarize(rec: Records, cfg: SimConfig) -> dict:
    lo, hi = cfg.warmup_s, cfg.warmup_s + cfg.duration_s
    inside = lambda t: lo <= t < hi  # noqa: E731
    reqs = [r for r in rec.requests if inside(r["arrival"])]
    ret = [r for r in reqs if r["turn"] > 0]
    n_all = sum(1 for t, _ in rec.arrivals if inside(t))  # unfinished requests count as SLO misses
    n_ret = sum(1 for t, turn in rec.arrivals if inside(t) and turn > 0)
    s = {
        "requests": n_all, "returning_requests": n_ret, "finished": len(reqs), "steps": rec.steps,
        "ttft_p50_s": _q([r["ttft"] for r in reqs], 0.5), "ttft_p99_s": _q([r["ttft"] for r in reqs], 0.99),
        "returning_ttft_p50_s": _q([r["ttft"] for r in ret], 0.5),
        "returning_ttft_p99_s": _q([r["ttft"] for r in ret], 0.99),
        "first_turn_ttft_p50_s": _q([r["ttft"] for r in reqs if r["turn"] == 0], 0.5),
    }
    for slo in SLOS_S:
        s[f"goodput_{slo:g}s"] = sum(r["ttft"] <= slo for r in reqs) / cfg.duration_s
        s[f"returning_goodput_{slo:g}s"] = sum(r["ttft"] <= slo for r in ret) / cfg.duration_s
        s[f"returning_slo_share_{slo:g}s"] = sum(r["ttft"] <= slo for r in ret) / n_ret if n_ret else None
    s["goodput"] = s[f"returning_goodput_{HEADLINE_SLO_S:g}s"]  # headline: returning turns at 5 s

    prompt = sum(r["prompt_tokens"] for r in ret) or 1
    s.update({
        "returning_gpu_hit_share": sum(r["gpu_hit"] for r in ret) / prompt,
        "returning_cpu_load_share": sum(r["cpu_load"] for r in ret) / prompt,
        "returning_recompute_share": sum(r["recomputed"] for r in ret) / prompt,
        "re_prefilled_tokens": sum(r["re_prefilled"] for r in reqs),
        "re_prefill_gpu_s": sum(r["re_prefill_s"] for r in reqs),
        "reloaded_tokens": sum(r["cpu_load"] for r in reqs),
    })

    pauses = [p for p in rec.pauses if inside(p["start"])]
    returned = [p for p in pauses if "kv_at_return" in p]
    for where in ("gpu", "cpu", "gone"):
        s[f"kv_at_return_{where}"] = sum(p["kv_at_return"] == where for p in returned) / len(returned) if returned else None
    predicted = [p for p in pauses if p["p10"] is not None and p["true_pause_s"] is not None]
    s["predictor_coverage"] = sum(p["confident"] for p in pauses) / len(pauses) if pauses else None
    s["predictor_in_p10_p90"] = (
        sum(p["p10"] <= p["true_pause_s"] <= p["p90"] for p in predicted) / len(predicted) if predicted else None)
    s["actions"] = {a: sum(p["action"] == a for p in pauses) for a in ("keep", "offload", "discard")}

    mem = [m for m in rec.memory if inside(m[0])]
    if mem:
        s["gpu_in_use_tokens_avg"] = sum(m[1] for m in mem) / len(mem) * BLOCK
        s["paused_kv_tokens_avg"] = sum(m[2] for m in mem) / len(mem) * BLOCK  # defines the tight cap
        s["gpu_used_tokens_avg"] = sum(m[3] for m in mem) / len(mem) * BLOCK

    warm = [w for w in rec.warmups if inside(w["issued"])]
    s["warmups_issued"] = len(warm)
    s["warmups_in_time"] = sum(bool(w.get("in_time")) for w in warm)
    s["returning_turns_warmed"] = sum(r["warmed"] for r in ret)
    return s


def write_outputs(out: Path, rec: Records, summary: dict, config: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("requests.csv", rec.requests), ("pauses.csv", rec.pauses)):
        if rows:
            keys = list(dict.fromkeys(k for r in rows for k in r))
            with (out / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "config.json").write_text(json.dumps(config, indent=2))
