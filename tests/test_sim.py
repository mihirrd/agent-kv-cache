import math

import pytest

from kvtier.cost import Calibration, CostModel, PredictivePolicy
from kvtier.sessions import BLOCK, ReplaySession
from kvtier.sim.engine import SimConfig, Simulator
from kvtier.sim.memory import CpuTier, GpuPool
from kvtier.sim.policies import LruPolicy, OraclePolicy, TtlPinPolicy
from kvtier.sim.server import Request, Server
from kvtier.sim.timing import StepModel

CAL = Calibration(
    prefill_points=((0, 0.0), (8000, 8.0), (16000, 32.0)),
    bandwidth_gb_s=20.0,
    reload_overhead_s=0.2,
    decode_base_s=0.02,
    decode_per_request_s=0.0,
    prefill_budget_tokens=2048,
)
COST = CostModel(CAL)


def toks(seed: int, n: int) -> list[int]:
    return [seed * 100_000 + i for i in range(n)]


def session(name: str, seed: int, prompt0: int, growth: int, pauses: list, outputs: int = 32) -> ReplaySession:
    """A session whose turn i prompt = turn i-1 prompt + `growth` tokens."""
    prompts = [toks(seed, prompt0 + i * growth) for i in range(len(pauses) + 1)]
    return ReplaySession.from_tokens(name, prompts, outputs=[outputs] * len(prompts), pauses=pauses + [None],
                                     programs=["tool"] * len(prompts))


def simulate(policy, sessions_at, cap_tokens, cpu=True, prefetch=False, duration=1000.0):
    cfg = SimConfig(load_per_min=1, cap_tokens=cap_tokens, cpu_tier=cpu, prefetch=prefetch, warmup_s=0,
                    duration_s=duration, drain_s=0, seed=0)
    sim = Simulator([s for _, s in sessions_at], policy, COST, cfg, shared_blocks=0)
    rec = sim.run(fixed_arrivals=sessions_at)
    return sim, rec


def by_turn(rec, prefix):
    return {r["turn"]: r for r in rec.requests if r["sid"].startswith(prefix)}


# --- timing -----------------------------------------------------------------------------------------
def test_prefill_cost_grows_with_context():
    steps = StepModel(COST)
    assert steps.prefill_s(16000, 1000) > steps.prefill_s(2000, 1000)
    assert steps.prefill_s(0, 4000) == pytest.approx(4.0)
    assert steps.reload_s(0) == 0.0
    assert steps.reload_s(8000) == pytest.approx(0.2 + 8000 * 57344 / 1e9 / 20)


# --- memory -----------------------------------------------------------------------------------------
def test_gpu_pool_hit_ownership_and_eviction():
    gpu = GpuPool(10)
    gpu.add([1, 2, 3], [None, "a", "a"])
    gpu.add([4, 5], ["b", "b"])
    assert gpu.hit([1, 2, 9, 3]) == 2  # stops at the first miss
    gpu.release([1, 2, 3], 0, now=5.0)
    gpu.release([4], 0, now=6.0)  # block 5 stays referenced
    assert gpu.cached_of("a") == 2 and gpu.cached_of("b") == 1
    assert gpu.evict_owner("b") == 1 and 5 in gpu.blocks  # in-use blocks survive
    assert gpu.evict_lru(5, owners=[None]) == 1 and 2 in gpu.blocks  # only the unowned block
    assert gpu.evict_owner("a") == 2 and gpu.used == 1


def test_cpu_tier_chunks_lru_and_owner_drop():
    cpu = CpuTier(capacity_tokens=2 * 256)
    shared = [("s", i) for i in range(16)]
    a = shared + [("a", i) for i in range(20)]  # 36 blocks: 2 full chunks, partial dropped
    cpu.store(a, "a", shared_blocks=16)
    assert cpu.hit_blocks(a) == 32
    b = [("b", i) for i in range(16)]
    cpu.store(b, "b", shared_blocks=0)  # third chunk: evicts the least recently used one
    assert len(cpu.chunks) == 2
    cpu.drop_owner("b")
    assert cpu.hit_blocks(b) == 0


# --- scheduler --------------------------------------------------------------------------------------
def test_scheduler_decodes_first_and_budget():
    gpu = GpuPool(10_000)
    srv = Server(gpu, None, StepModel(COST), 0, make_room=lambda n, t: 0)
    srv.submit(Request("a", 0, [], 20, 3, 0.0))
    srv.submit(Request("b", 0, [], 5000, 3, 0.0))
    p1 = srv.plan(0.0)
    assert [(r.sid, k) for r, k in p1.chunks] == [("a", 20), ("b", 2028)]  # budget 2048
    srv.apply(p1, 1.0)
    p2 = srv.plan(1.0)
    assert [r.sid for r in p2.decodes] == ["a"] and p2.chunks[0][1] == 2048
    assert p2.duration == pytest.approx(0.02 + StepModel(COST).prefill_s(2028, 2048))


# --- end-to-end scenarios ---------------------------------------------------------------------------
def test_no_pressure_turn_reuses_previous_prompt():
    s = session("a", 1, prompt0=1000, growth=200, pauses=[3.0, 3.0])
    _, rec = simulate(LruPolicy(), [(0.0, s)], cap_tokens=50_000, cpu=False)
    t = by_turn(rec, "a")
    assert t[1]["gpu_hit"] == 1000 // BLOCK * BLOCK and t[1]["re_prefilled"] == 0
    assert t[2]["gpu_hit"] == 1200 // BLOCK * BLOCK
    assert t[1]["arrival"] == pytest.approx(t[0]["finish"] + 3.0)


def two_sessions():
    # a pauses long, b arrives during a's pause and needs a's memory
    a = session("a", 1, prompt0=3000, growth=100, pauses=[60.0])
    b = session("b", 2, prompt0=3000, growth=100, pauses=[1.0])
    return [(0.0, a), (10.0, b)]


def test_lru_without_cpu_recomputes_and_with_cpu_reloads():
    cap = 5000  # one session's prompt + output (~3,050 tokens) fits, two do not
    _, rec = simulate(LruPolicy(), two_sessions(), cap_tokens=cap, cpu=False)
    a1 = by_turn(rec, "a")[1]
    assert a1["gpu_hit"] == 0 and a1["re_prefilled"] > 2900
    _, rec = simulate(LruPolicy(), two_sessions(), cap_tokens=cap, cpu=True)
    a1 = by_turn(rec, "a")[1]
    assert a1["cpu_load"] >= 2816 and a1["re_prefilled"] < 256  # whole 256-token chunks reload


def test_oracle_evicts_the_long_pause_first():
    from kvtier.policy import Session

    pol = OraclePolicy(COST)
    paused = [Session(name, context_tokens=2000, pause_start=0.0, true_pause_s=p)
              for name, p in (("short", 30.0), ("long", 200.0), ("ended", None), ("mid", 100.0))]
    for s in paused:
        pol.on_pause(s, 0.0)
    assert [s.session_id for s in pol.eviction_order(paused, now=1.0)] == ["ended", "long", "mid", "short"]


def test_lru_session_returns_to_cpu_copy():
    _, rec = simulate(LruPolicy(), two_sessions(), cap_tokens=5000, cpu=True)
    at_return = {p["sid"].split("#")[0]: p.get("kv_at_return") for p in rec.pauses if p["turn"] == 0}
    assert at_return["a"] == "cpu" and at_return["b"] == "gpu"


def test_ttl_pin_orders_expired_first():
    from kvtier.policy import Session

    pol = TtlPinPolicy(ttl_s=5.0)
    old_pinned = Session("p", pause_start=98.0, last_used=1.0)
    expired = Session("e", pause_start=90.0, last_used=50.0)
    assert [s.session_id for s in pol.eviction_order([old_pinned, expired], now=100.0)] == ["e", "p"]


def test_prefetch_warms_offloaded_session():
    a = session("a", 1, prompt0=3000, growth=100, pauses=[60.0])
    b = session("b", 2, prompt0=3000, growth=100, pauses=[1.0])
    sim, rec = simulate(OraclePolicy(COST), [(0.0, a), (10.0, b)], cap_tokens=20_000, cpu=True, prefetch=True)
    a1 = by_turn(rec, "a")[1]
    assert rec.warmups and rec.warmups[0]["in_time"]
    assert a1["warmed"] and a1["gpu_hit"] >= 2992 and a1["cpu_load"] == 0


# --- invariants -------------------------------------------------------------------------------------
def many_sessions():
    return [session(f"s{i}", i, prompt0=1500 + 300 * i, growth=250, pauses=[0.5 * i + 0.1, 8.0, 0.05, 20.0])
            for i in range(8)]


@pytest.mark.parametrize("policy", ["vllm", "lru-cpu", "ttl", "predictive", "oracle"])
def test_invariants_under_pressure(policy):
    pols = {"vllm": LruPolicy(), "lru-cpu": LruPolicy(), "ttl": TtlPinPolicy(5.0),
            "predictive": PredictivePolicy(COST), "oracle": OraclePolicy(COST)}
    arrivals = [(5.0 * i, s) for i, s in enumerate(many_sessions())]
    sim, rec = simulate(pols[policy], arrivals, cap_tokens=12_000, cpu=policy != "vllm", prefetch=True)
    assert len(rec.requests) == sum(len(s.turns) for _, s in arrivals)  # every turn finished
    assert sim.gpu.anon == 0 and all(b.refs == 0 for b in sim.gpu.blocks.values())
    assert sim.gpu.used <= sim.gpu.capacity
    for r in rec.requests:
        assert r["gpu_hit"] + r["cpu_load"] + r["recomputed"] == r["prompt_tokens"]
        assert r["first_token"] >= r["arrival"] and r["finish"] >= r["first_token"]


def test_same_seed_same_result():
    from kvtier.sim.metrics import summarize

    sessions = many_sessions()
    cfg = SimConfig(load_per_min=6, cap_tokens=20_000, warmup_s=10, duration_s=120, drain_s=60, seed=3)
    runs = [Simulator(sessions, LruPolicy(), COST, cfg, 0).run().requests for _ in range(2)]
    assert runs[0] == runs[1] and runs[0]
    assert summarize(Simulator(sessions, LruPolicy(), COST, cfg, 0).run(), cfg)["requests"] > 0


def test_no_pressure_policies_agree():
    arrivals = [(5.0 * i, s) for i, s in enumerate(many_sessions())]
    ttft = {}
    for name, pol, cpu in (("vllm", LruPolicy(), False), ("lru-cpu", LruPolicy(), True), ("ttl", TtlPinPolicy(5), True)):
        _, rec = simulate(pol, arrivals, cap_tokens=500_000, cpu=cpu)
        ttft[name] = [round(r["ttft"], 9) for r in sorted(rec.requests, key=lambda r: (r["sid"], r["turn"]))]
    assert ttft["vllm"] == ttft["lru-cpu"] == ttft["ttl"]


def test_long_call_offload_classifies_by_predicted_median():
    from kvtier.policy import Action, Session
    from kvtier.predictor import PausePredictor
    from kvtier.sim.policies import LongCallOffloadPolicy

    pred = PausePredictor(min_samples=5)
    for _ in range(5):
        pred.observe("x", "pip", 12.0)
        pred.observe("x", "grep", 0.05)
    pol = LongCallOffloadPolicy(threshold_s=2.0, predictor=pred)
    assert pol.on_pause(Session("s", program="pip"), 0.0) is Action.OFFLOAD
    assert pol.on_pause(Session("s", program="grep"), 0.0) is Action.KEEP
    assert pol.prefetch_at(Session("s")) is None  # no prefetch
    oracle = LongCallOffloadPolicy(threshold_s=2.0, oracle=True)
    assert oracle.on_pause(Session("s", program="grep", true_pause_s=30.0), 0.0) is Action.OFFLOAD
    assert oracle.on_pause(Session("s", program="pip", true_pause_s=0.5), 0.0) is Action.KEEP
    assert oracle.on_pause(Session("s", true_pause_s=None), 0.0) is Action.OFFLOAD  # session ended


def test_long_call_offload_runs_end_to_end():
    from kvtier.sim.policies import LongCallOffloadPolicy

    arrivals = [(5.0 * i, s) for i, s in enumerate(many_sessions())]
    sim, rec = simulate(LongCallOffloadPolicy(2.0, oracle=True), arrivals, cap_tokens=12_000, cpu=True)
    assert len(rec.requests) == sum(len(s.turns) for _, s in arrivals)
    assert {p["action"] for p in rec.pauses} == {"keep", "offload"}
    assert not rec.warmups


def test_remaining_time_eviction_order():
    from kvtier.policy import Session
    from kvtier.predictor import PausePredictor
    from kvtier.sim.policies import RemainingTimeEviction

    pred = PausePredictor(min_samples=5)
    for _ in range(5):
        pred.observe("x", "grep", 0.05)
        pred.observe("x", "pip", 20.0)
    pol = RemainingTimeEviction(pred)
    grep_fresh = Session("grep_fresh", program="grep", pause_start=99.99, last_used=99.99)
    grep_overdue = Session("grep_overdue", program="grep", pause_start=90.0, last_used=90.0)  # outlasted all greps
    pip_mid = Session("pip_mid", program="pip", pause_start=95.0, last_used=95.0)  # ~15 s left
    order = [s.session_id for s in pol.eviction_order([grep_fresh, pip_mid, grep_overdue], now=100.0)]
    assert order == ["grep_overdue", "pip_mid", "grep_fresh"]
    belady = RemainingTimeEviction(oracle=True)
    a = Session("a", pause_start=0.0, true_pause_s=50.0)
    b = Session("b", pause_start=0.0, true_pause_s=None)  # session ended: never returns
    c = Session("c", pause_start=0.0, true_pause_s=5.0)
    assert [s.session_id for s in belady.eviction_order([c, a, b], now=1.0)] == ["b", "a", "c"]


def test_continuum_ttl_choice():
    from kvtier.policy import Session
    from kvtier.sim.policies import ContinuumPolicy

    pol = ContinuumPolicy(COST, k_min=3)
    s = Session("s", program="pytest", context_tokens=8000)  # gain = 8.0 s prefill - 0.2229 s reload
    gain = 8.0 - COST.reload_s(8000)
    for x in (1.0, 2.0, 3.0, 50.0, 60.0):  # 5 > k_min records: this tool's own history
        pol.record("pytest", x)
    # candidates: tau=1 -> 0.2 gain - 1, 2 -> 0.4 gain - 2, 3 -> 0.6 gain - 3 (best), 50 -> 0.8 gain - 50, ...
    assert pol.ttl(s) == 3.0 and 0.6 * gain - 3 > 0.4 * gain - 2
    pol.record("grep", 0.05)  # too few grep records: falls back to everything recorded
    assert pol.ttl(Session("g", program="grep", context_tokens=8000)) == 3.0
    cold = ContinuumPolicy(COST)  # no history: Exp(1) default, tau = ln(gain)
    assert cold.ttl(s) == pytest.approx(math.log(gain))
    cold.observe_queue_delay(2.0)
    assert cold.queue_delay_s == pytest.approx(0.1)  # running average starts from zero


def test_continuum_pins_then_evicts_latest_program_first():
    from kvtier.policy import Session
    from kvtier.sim.policies import ContinuumPolicy

    pol = ContinuumPolicy(COST)
    expired = Session("expired", pinned_until=5.0, last_used=1.0, program_arrival=0.0)
    pinned_old = Session("pinned_old", pinned_until=50.0, last_used=8.0, program_arrival=1.0)
    pinned_new = Session("pinned_new", pinned_until=50.0, last_used=9.0, program_arrival=7.0)
    order = pol.eviction_order([pinned_old, pinned_new, expired], now=10.0)
    assert [s.session_id for s in order] == ["expired", "pinned_new", "pinned_old"]


def test_program_fcfs_queue_order():
    gpu = GpuPool(10_000)
    srv = Server(gpu, None, StepModel(COST), 0, make_room=lambda n, t: 0,
                 queue_key=lambda r: (not r.pinned, r.program_arrival, r.arrival))
    srv.submit(Request("new", 0, [], 100, 2, 5.0, program_arrival=5.0))
    srv.submit(Request("old", 3, [], 100, 2, 6.0, program_arrival=1.0))
    srv.submit(Request("pinned", 2, [], 100, 2, 7.0, program_arrival=3.0, pinned=True))
    plan = srv.plan(7.0)
    assert [r.sid for r in plan.admitted] == ["pinned", "old", "new"]


def test_continuum_end_to_end():
    from kvtier.sim.policies import ContinuumPolicy

    arrivals = [(5.0 * i, s) for i, s in enumerate(many_sessions())]
    cfg = SimConfig(load_per_min=1, cap_tokens=12_000, cpu_tier=True, prefetch=False, warmup_s=0,
                    duration_s=1000, drain_s=0, seed=0, program_fcfs=True)
    sim = Simulator([s for _, s in arrivals], ContinuumPolicy(COST), COST, cfg, shared_blocks=0)
    rec = sim.run(fixed_arrivals=arrivals)
    assert len(rec.requests) == sum(len(s.turns) for _, s in arrivals)
    assert sim.gpu.anon == 0 and all(b.refs == 0 for b in sim.gpu.blocks.values())
