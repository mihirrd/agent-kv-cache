"""C3 cost model and the predictive placement policy."""

from dataclasses import dataclass

from kvtier.policy import Action, KVPolicy, Session, State
from kvtier.predictor import PausePredictor

KV_BYTES_PER_TOKEN_QWEN7B = 28 * 4 * 128 * 2 * 2  # layers x KV heads x head dim x (K, V) x fp16 = 57,344


@dataclass(frozen=True)
class Calibration:
    """Measured on gpubox (week 2 calibration replaces the placeholders)."""

    prefill_points: tuple[tuple[int, float], ...]  # (prompt tokens, cold prefill seconds), ascending
    bandwidth_gb_s: float  # CPU <-> GPU KV transfer
    reload_overhead_s: float = 0.0  # fixed per-reload cost beyond the transfer
    kv_bytes_per_token: int = KV_BYTES_PER_TOKEN_QWEN7B


# Placeholder from the Gate A measurements (7B-AWQ, TP=2, TRITON_ATTN, 2026-10-20).
GATE_A_TP2 = Calibration(
    prefill_points=((0, 0.0), (8000, 7.6), (16000, 29.2), (28000, 90.0)),
    bandwidth_gb_s=20.0,
    reload_overhead_s=0.25,
)


class CostModel:
    def __init__(self, cal: Calibration):
        self.cal = cal

    def kv_gb(self, tokens: int) -> float:
        return tokens * self.cal.kv_bytes_per_token / 1e9

    def recompute_s(self, tokens: int) -> float:
        """Cold prefill time, linear between calibration points and along the last segment beyond them."""
        pts = self.cal.prefill_points
        for (t0, s0), (t1, s1) in zip(pts, pts[1:]):
            if tokens <= t1:
                return s0 + (s1 - s0) * (tokens - t0) / (t1 - t0)
        (t0, s0), (t1, s1) = pts[-2], pts[-1]
        return s1 + (s1 - s0) * (tokens - t1) / (t1 - t0)

    def transfer_s(self, tokens: int) -> float:
        return self.kv_gb(tokens) / self.cal.bandwidth_gb_s

    def reload_s(self, tokens: int) -> float:
        return self.cal.reload_overhead_s + self.transfer_s(tokens)

    def offload_round_trip_s(self, tokens: int) -> float:
        return self.transfer_s(tokens) + self.reload_s(tokens)


class PredictivePolicy(KVPolicy):
    """Tool-aware placement: predicted pause vs offload/recompute cost (SPEC.md C3, C4)."""

    def __init__(self, cost: CostModel, predictor: PausePredictor | None = None, min_remaining_s: float = 0.1):
        self.cost = cost
        self.predictor = predictor or PausePredictor()
        self.min_remaining_s = min_remaining_s

    def on_pause(self, s: Session, now: float) -> Action:
        s.prediction = self.predictor.predict(s.session_id, s.program)
        if s.prediction is None or not s.prediction.confident:
            return Action.KEEP  # unknown pause: behave like vLLM and let eviction_order decide
        if s.prediction.p90 < self.cost.offload_round_trip_s(s.context_tokens):
            return Action.KEEP
        if self.cost.recompute_s(s.context_tokens) < self.cost.reload_s(s.context_tokens):
            return Action.DISCARD
        return Action.OFFLOAD

    def score(self, s: Session, now: float) -> float:
        """Recompute seconds saved per GB-second of GPU memory held; lowest is evicted first."""
        remaining = s.prediction.p50 - (now - s.pause_start)
        return self.cost.recompute_s(s.context_tokens) / (
            max(self.cost.kv_gb(s.context_tokens), 1e-9) * max(remaining, self.min_remaining_s)
        )

    def in_return_window(self, s: Session, now: float) -> bool:
        """Confident session whose predicted return (p10) is closer than a reload: evicting it now would hurt."""
        return now >= s.pause_start + s.prediction.p10 - self.cost.reload_s(s.context_tokens)

    def eviction_order(self, paused: list[Session], now: float) -> list[Session]:
        # Low-confidence sessions first in LRU order (vLLM's default), then confident ones by score;
        # sessions inside their return window are exempt and go last.
        unsure = sorted((s for s in paused if not (s.prediction and s.prediction.confident)), key=lambda s: s.last_used)
        sure = [s for s in paused if s.prediction and s.prediction.confident]
        exempt = [s for s in sure if self.in_return_window(s, now)]
        ranked = sorted((s for s in sure if s not in exempt), key=lambda s: self.score(s, now))
        return unsure + ranked + sorted(exempt, key=lambda s: self.score(s, now))

    def prefetch_at(self, s: Session) -> float | None:
        if s.state is not State.OFFLOADED or not (s.prediction and s.prediction.confident):
            return None
        return s.pause_start + s.prediction.p10 - self.cost.reload_s(s.context_tokens)

    def on_return(self, s: Session, pause_s: float) -> None:
        self.predictor.observe(s.session_id, s.program, pause_s)
