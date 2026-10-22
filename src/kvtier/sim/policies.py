"""Baseline policies for the simulator; the predictive policy lives in kvtier.cost."""

import math
import random

from kvtier.cost import CostModel, PredictivePolicy
from kvtier.policy import Action, KVPolicy, Session
from kvtier.predictor import PausePredictor, Prediction


class LruPolicy(KVPolicy):
    """vLLM's default: keep everything, evict the least recently used session first.
    Used for both 'vllm' (CPU tier off) and 'always-offload' (LMCache on)."""

    def on_pause(self, s: Session, now: float) -> Action:
        return Action.KEEP

    def eviction_order(self, paused: list[Session], now: float) -> list[Session]:
        return sorted(paused, key=lambda s: s.last_used)


class TtlPinPolicy(LruPolicy):
    """Continuum-style: a paused session is pinned for a fixed TTL, then falls back to LRU."""

    def __init__(self, ttl_s: float):
        self.ttl_s = ttl_s

    def eviction_order(self, paused: list[Session], now: float) -> list[Session]:
        expired = [s for s in paused if now - s.pause_start >= self.ttl_s]
        pinned = [s for s in paused if now - s.pause_start < self.ttl_s]
        return sorted(expired, key=lambda s: s.last_used) + sorted(pinned, key=lambda s: s.last_used)


class OraclePolicy(PredictivePolicy):
    """C3 rules fed the true pause (x (1 + e), e uniform in +-noise); ended sessions pause forever."""

    def __init__(self, cost: CostModel, noise: float = 0.0, seed: int = 0, **kwargs):
        super().__init__(cost, PausePredictor(), **kwargs)
        self.noise = noise
        self.rng = random.Random(seed)

    def predict(self, s: Session) -> Prediction:
        p = math.inf if s.true_pause_s is None else s.true_pause_s
        if self.noise and math.isfinite(p):
            p *= 1 + self.rng.uniform(-self.noise, self.noise)
        return Prediction(p, p, p, 1, True, ("oracle",))

    def on_return(self, s: Session, pause_s: float) -> None:
        pass


class EvictOnlyMixin:
    """Use the prediction only to order evictions under pressure: never offload or discard on pause.
    Diagnostic for Gate C: separates the value of eviction order from proactive offload + prefetch."""

    def on_pause(self, s: Session, now: float) -> Action:
        super().on_pause(s, now)  # sets s.prediction
        return Action.KEEP


class EvictOnlyOracle(EvictOnlyMixin, OraclePolicy):
    pass


class EvictOnlyPredictive(EvictOnlyMixin, PredictivePolicy):
    pass


class LongCallOffloadPolicy(LruPolicy):
    """Classify each tool call as long or short when its pause starts; offload long ones at once,
    keep short ones on GPU. Eviction under pressure is LRU, as in always-offload; no prefetch.

    Long = predicted median pause > threshold_s (from the C2 predictor), or with `oracle=True`
    the true pause > threshold_s (an ended session counts as long).
    """

    def __init__(self, threshold_s: float, predictor: PausePredictor | None = None, oracle: bool = False):
        self.threshold_s = threshold_s
        self.predictor = predictor or PausePredictor()
        self.oracle = oracle

    def is_long(self, s: Session) -> bool:
        if self.oracle:
            p = math.inf if s.true_pause_s is None else s.true_pause_s
            s.prediction = Prediction(p, p, p, 1, True, ("oracle",))
            return p > self.threshold_s
        s.prediction = self.predictor.predict(s.session_id, s.program)
        return s.prediction is not None and s.prediction.p50 > self.threshold_s

    def on_pause(self, s: Session, now: float) -> Action:
        return Action.OFFLOAD if self.is_long(s) else Action.KEEP

    def on_return(self, s: Session, pause_s: float) -> None:
        if not self.oracle:
            self.predictor.observe(s.session_id, s.program, pause_s)


class RemainingTimeEviction(LruPolicy):
    """Caching policy: keep everything while there is room; when space runs out, evict the paused
    session expected to stay away longest. Expected time left comes from past pauses of the same
    tool that lasted longer than this pause has so far (C2 back-off keys). Ties and sessions with no
    history fall back to LRU. With `oracle=True` it uses the true time left (Belady's rule)."""

    def __init__(self, predictor: PausePredictor | None = None, oracle: bool = False):
        self.predictor = predictor or PausePredictor()
        self.oracle = oracle

    def time_left(self, s: Session, now: float) -> float:
        elapsed = now - s.pause_start
        if self.oracle:
            return math.inf if s.true_pause_s is None else s.true_pause_s - elapsed
        left = self.predictor.remaining(s.session_id, s.program, elapsed)
        return 0.0 if left is None else left

    def eviction_order(self, paused: list[Session], now: float) -> list[Session]:
        return sorted(paused, key=lambda s: (-self.time_left(s, now), s.last_used))

    def on_return(self, s: Session, pause_s: float) -> None:
        if not self.oracle:
            self.predictor.observe(s.session_id, s.program, pause_s)


def warm_start(policy, sessions) -> None:
    """Seed the predictor with the pauses of training sessions (as other, earlier sessions)."""
    for rs in sessions:
        for t in rs.turns:
            if t.pause_s is not None:
                policy.predictor.observe(f"train:{rs.session_id}", t.program, t.pause_s)
