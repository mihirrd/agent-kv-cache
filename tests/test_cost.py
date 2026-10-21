import pytest

from kvtier.cost import Calibration, CostModel, PredictivePolicy
from kvtier.policy import Action, Session, State
from kvtier.predictor import PausePredictor

CAL = Calibration(
    prefill_points=((0, 0.0), (8000, 8.0), (16000, 32.0)),
    bandwidth_gb_s=20.0,
    reload_overhead_s=0.2,
    kv_bytes_per_token=50_000,
)


@pytest.fixture
def cost():
    return CostModel(CAL)


def test_recompute_interpolates_and_extrapolates(cost):
    assert cost.recompute_s(4000) == pytest.approx(4.0)
    assert cost.recompute_s(12000) == pytest.approx(20.0)
    assert cost.recompute_s(20000) == pytest.approx(44.0)  # last segment: 3 ms/token


def test_transfer_and_reload(cost):
    assert cost.kv_gb(8000) == pytest.approx(0.4)
    assert cost.transfer_s(8000) == pytest.approx(0.02)
    assert cost.reload_s(8000) == pytest.approx(0.22)
    assert cost.offload_round_trip_s(8000) == pytest.approx(0.24)


def policy_with(pauses_by_program, **kwargs):
    predictor = PausePredictor(min_samples=5)
    for program, pauses in pauses_by_program.items():
        for p in pauses:
            predictor.observe("other", program, p)
    return PredictivePolicy(CostModel(CAL), predictor, **kwargs)


def paused(program, tokens=8000, start=100.0, sid="s", last_used=None):
    return Session(sid, context_tokens=tokens, state=State.PAUSED, program=program, pause_start=start,
                   last_used=start if last_used is None else last_used)


def test_on_pause_decisions():
    pol = policy_with({"grep": [0.05] * 5, "pytest": [8.0, 9.0, 10.0, 11.0, 12.0], "make": [0.01, 0.02, 5, 50, 60]})
    assert pol.on_pause(paused("grep"), 100.0) is Action.KEEP  # p90 below the 0.24 s round trip
    assert pol.on_pause(paused("pytest"), 100.0) is Action.OFFLOAD  # 8 s recompute > 0.22 s reload
    assert pol.on_pause(paused("pytest", tokens=100), 100.0) is Action.DISCARD  # 0.1 s recompute < 0.2 s reload
    assert pol.on_pause(paused("make"), 100.0) is Action.KEEP  # low confidence -> default behaviour
    assert pol.on_pause(paused("unseen"), 100.0) is Action.KEEP  # global key: mixed pauses, not confident
    s = paused("pytest")
    pol.on_pause(s, 100.0)
    assert s.prediction.key == ("program", "pytest")


def test_eviction_order():
    pol = policy_with({"pytest": [30.0] * 5, "cargo": [300.0] * 5, "make": [0.01, 0.02, 5, 50, 60]})
    short = paused("pytest", sid="short")
    long = paused("cargo", sid="long")
    unsure_old = paused("make", sid="unsure_old", last_used=50.0)
    unsure_new = paused("make", sid="unsure_new", last_used=90.0)
    for s in (short, long, unsure_old, unsure_new):
        pol.on_pause(s, 100.0)
    order = [s.session_id for s in pol.eviction_order([short, unsure_new, long, unsure_old], now=101.0)]
    # Unsure sessions first by LRU; then the long pause (holds memory longest per second saved).
    assert order == ["unsure_old", "unsure_new", "long", "short"]
    # At t=129.9 the pytest session is inside its return window (p10 30 s - 0.22 s reload) and goes last.
    assert pol.in_return_window(short, 129.9)
    assert [s.session_id for s in pol.eviction_order([short, long], now=129.9)] == ["long", "short"]


def test_score_uses_remaining_p50_with_floor():
    pol = policy_with({"pytest": [30.0] * 5}, min_remaining_s=0.1)
    s = paused("pytest")
    pol.on_pause(s, 100.0)
    # recompute 8 s / (0.4 GB x 20 s remaining)
    assert pol.score(s, 110.0) == pytest.approx(8.0 / (0.4 * 20.0))
    # overdue session: remaining floored at 0.1 s
    assert pol.score(s, 200.0) == pytest.approx(8.0 / (0.4 * 0.1))


def test_prefetch_only_for_confident_offloaded():
    pol = policy_with({"pytest": [10.0] * 5})
    s = paused("pytest")
    pol.on_pause(s, 100.0)
    assert pol.prefetch_at(s) is None  # still on GPU
    s.state = State.OFFLOADED
    assert pol.prefetch_at(s) == pytest.approx(100.0 + 10.0 - 0.22)


def test_on_return_feeds_predictor():
    pol = policy_with({})
    s = paused("cargo", sid="x")
    for _ in range(5):
        pol.on_return(s, 42.0)
    assert pol.predictor.predict("x", "cargo").p50 == 42.0
