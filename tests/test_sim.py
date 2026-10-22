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
