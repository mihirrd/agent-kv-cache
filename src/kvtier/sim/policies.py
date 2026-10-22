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


class ContinuumPolicy(LruPolicy):
    """Continuum (Li et al., arXiv:2511.02230): pin a paused session's KV for a TTL chosen at pause start.

    TTL = argmax over tau in {0} + past durations of this tool of  P(pause <= tau) x (T x eta + prefill - reload) - tau,
    with P the empirical CDF of the tool's past durations (all tools' if it has <= k_min records; an Exp(1)
    default if there is almost no history), T a running average of queueing delay, and eta = 1 (their
    memoryfulness factor; -Corr(k, N - k) is 1 within one program). Pinned KV is not evicted unless
    admission is stuck; then pinned sessions go latest program arrival first. Expired pins become ordinary
    LRU cache with the LMCache CPU copy. Their queue order (pinned first, then program arrival) is
    SimConfig.program_fcfs.
    """

    def __init__(self, cost: CostModel, eta: float = 1.0, k_min: int = 100, queue_ema: float = 0.05):
        self.cost = cost
        self.eta = eta
        self.k_min = k_min
        self.queue_ema = queue_ema
        self.by_tool: dict[str, list[float]] = {}
        self.everything: list[float] = []
        self.queue_delay_s = 0.0  # T, initialised to zero as in the paper

    def observe_queue_delay(self, delay_s: float) -> None:
        self.queue_delay_s += self.queue_ema * (delay_s - self.queue_delay_s)

    def record(self, program: str, pause_s: float) -> None:
        self.by_tool.setdefault(program, []).append(pause_s)
        self.everything.append(pause_s)

    def ttl(self, s: Session) -> float:
        gain = (self.queue_delay_s * self.eta + self.cost.recompute_s(s.context_tokens)
                - self.cost.reload_s(s.context_tokens))
        history = self.by_tool.get(s.program, [])
        if len(history) <= self.k_min:
            history = self.everything
        if len(history) < 5:  # cold start: durations ~ Exp(1 s); maximise (1 - e^-tau) gain - tau
            return math.log(gain) if gain > 1 else 0.0
        xs = sorted(history)
        n = len(xs)
        best_tau, best = 0.0, 0.0
        for i, tau in enumerate(xs):
            value = (i + 1) / n * gain - tau
            if value > best:
                best_tau, best = tau, value
        return best_tau

    def on_pause(self, s: Session, now: float) -> Action:
        s.pinned_until = now + self.ttl(s)
        return Action.KEEP

    def eviction_order(self, paused: list[Session], now: float) -> list[Session]:
        free = sorted((s for s in paused if s.pinned_until <= now), key=lambda s: s.last_used)
        pinned = sorted((s for s in paused if s.pinned_until > now), key=lambda s: -s.program_arrival)
        return free + pinned

    def on_return(self, s: Session, pause_s: float) -> None:
        self.record(s.program, pause_s)


def warm_start(policy, sessions) -> None:
    """Seed a policy's pause history with the training sessions (as other, earlier sessions)."""
    for rs in sessions:
        for t in rs.turns:
            if t.pause_s is None:
                continue
            if isinstance(policy, ContinuumPolicy):
                policy.record(t.program, t.pause_s)
            else:
                policy.predictor.observe(f"train:{rs.session_id}", t.program, t.pause_s)
