import asyncio
import json
from pathlib import Path

import httpx
import pytest

from kvtier.cost import GATE_A_TP2, CostModel
from kvtier.replay import (
    Replayer,
    ReplayTrace,
    RunConfig,
    arrival_schedule,
    attach_messages,
    check,
    load_traces,
    parse_metrics,
    summarize,
)
from kvtier.sessions import ReplaySession
from kvtier.sim.engine import SimConfig, Simulator
from kvtier.sim.policies import LruPolicy

SESSIONS = Path("traces/sessions/verified.jsonl")
BASE = "http://localhost:8000"


def trace(name: str, n_turns: int, pause: float = 1.0) -> ReplayTrace:
    """Turn i's prompt is the first 2 + 2i messages: system, task, then (reply, tool output) pairs."""
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": f"task {name}"}]
    for i in range(n_turns):
        messages += [{"role": "assistant", "content": f"cmd {i}"}, {"role": "user", "content": f"out {i}"}]
    prompts = [list(range(20 + 10 * i)) for i in range(n_turns)]
    session = ReplaySession.from_tokens(name, prompts, outputs=[5 + i for i in range(n_turns)],
                                        pauses=[pause] * (n_turns - 1) + [None])
    return ReplayTrace(session, messages, [2 + 2 * i for i in range(n_turns)])


# --- arrivals ----------------------------------------------------------------------------------------
def test_arrival_schedule_matches_simulator():
    sessions = [trace(f"s{k}", 2).session for k in range(5)]
    cfg = SimConfig(load_per_min=30, cap_tokens=100_000, warmup_s=0, duration_s=120, drain_s=0, seed=3)
    sim = Simulator(sessions, LruPolicy(), CostModel(GATE_A_TP2), cfg, shared_blocks=0)
    rec = sim.run()
    schedule = arrival_schedule(len(sessions), 30, 120, seed=3)
    assert len(schedule) > 20
    assert [t for t, _, _ in schedule] == pytest.approx([t for t, turn in rec.arrivals if turn == 0])
    assert [sessions[i].session_id for _, i, _ in schedule] == [sid.split("#")[0] for sid in sim.live]


# --- prompts -----------------------------------------------------------------------------------------
def test_attach_messages_and_salt(tmp_path):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "task"},
            {"role": "assistant", "content": "a1"}, {"role": "user", "content": "o1"},
            {"role": "assistant", "content": "a2"}, {"role": "user", "content": "o2"}]
    lines = [{"type": "header"}, {"type": "turn", "prompt_messages": 2}, {"type": "turn", "prompt_messages": 4},
             {"type": "turn", "prompt_messages": 6}, {"type": "footer", "messages": msgs}]
    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    session = ReplaySession.from_tokens("t", [[1] * 16, [1] * 32])  # third turn dropped by the context cap
    tr = attach_messages(session, path)
    assert tr.prompt_messages == [2, 4]
    assert tr.prompt(1) == msgs[:4]
    salted = tr.prompt(1, "(replay 1)\n")
    assert salted[0] == msgs[0]  # system prompt still shared between copies
    assert salted[1]["content"] == "(replay 1)\ntask"
    assert salted[2:] == msgs[2:4]
    assert msgs[1]["content"] == "task"  # the recording is not modified


def test_parse_metrics():
    text = ('# HELP vllm:num_requests_running x\n'
            'vllm:num_requests_running{engine="0",model_name="m"} 3.0\n'
            'vllm:prefix_cache_hits_total{engine="0"} 100.0\n'
            'vllm:prefix_cache_hits_total{engine="1"} 20.0\n'
            'vllm:prefix_cache_queries_total 400.0\n')
    m = parse_metrics(text)
    assert m["num_requests_running"] == 3.0
    assert m["prefix_cache_hits_total"] == 120.0
    assert m["prefix_cache_queries_total"] == 400.0
    assert "num_requests_waiting" not in m


# --- full replay against a fake server ----------------------------------------------------------------
class FakeServer:
    def __init__(self):
        self.bodies = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text="vllm:num_requests_running 1.0\n")
        body = json.loads(request.content)
        self.bodies.append(body)
        n_prompt = 3 + sum(len(m["content"]) for m in body["messages"])
        chunks = [{"choices": [{"delta": {"role": "assistant", "content": ""}}]},
                  {"choices": [{"delta": {"content": "x"}}]},
                  {"choices": [], "usage": {"prompt_tokens": n_prompt, "completion_tokens": body["max_tokens"],
                                            "prompt_tokens_details": {"cached_tokens": 16}}}]
        sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=sse)


def replay_fake(traces, **kw):
    server = FakeServer()
    cfg = RunConfig(load_per_min=600, warmup_s=0, duration_s=0.5, drain_s=2, pause_scale=0.01, scrape_every_s=0.1, **kw)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(server), base_url=cfg.base_url) as client:
            rp = Replayer(traces, cfg, client, "fake-model")
            await rp.run()
            return rp

    return server, asyncio.run(go()), cfg


def record_for(rp, body) -> dict:
    """The record of the request with this body (turn i's prompt has 2 + 2i messages)."""
    sid, turn = body["vllm_xargs"]["session_id"], (len(body["messages"]) - 2) // 2
    return next(r for r in rp.records if r["sid"] == sid and r["turn"] == turn)


def test_replay_sends_recorded_prompts_with_forced_output():
    traces = [trace("a", 3), trace("b", 2)]
    server, rp, cfg = replay_fake(traces)
    assert rp.records and all(r["ttft"] is not None and r["finish"] is not None for r in rp.records)
    assert len(server.bodies) == len(rp.records)
    by_trace = {tr.session_id: tr for tr in traces}
    for body in server.bodies:
        r = record_for(rp, body)
        assert body["min_tokens"] == body["max_tokens"] == r["output_tokens"]
        assert body["temperature"] == 0 and body["stream"] is True
        assert body["vllm_xargs"] == {"session_id": r["sid"]}
        assert "priority" not in body
        tr = by_trace[r["trace"]]
        expected = tr.prompt(r["turn"])
        assert len(body["messages"]) == len(expected)
        assert body["messages"][1]["content"].startswith("(replay ")
        assert body["messages"][1]["content"].endswith(expected[1]["content"])
        assert body["messages"][2:] == expected[2:]
    # every session copy replays all its turns, in order
    for sid in {r["sid"] for r in rp.records}:
        turns = [r["turn"] for r in rp.records if r["sid"] == sid]
        assert turns == list(range(len(by_trace[sid.split("#")[0]].session.turns)))
    assert rp.metrics and rp.metrics[0]["num_requests_running"] == 1.0
    s = summarize(rp.records, cfg)
    assert s["errors"] == 0 and s["output_length_mismatches"] == 0
    assert s["served"] == s["requests"] > 0


def test_replay_priority_by_session_arrival():
    server, rp, _ = replay_fake([trace("a", 2)], priority="session-arrival")
    for body in server.bodies:
        assert body["priority"] == int(record_for(rp, body)["session_arrival"] * 1000)
    # every turn of a session keeps the priority of its arrival
    for sid in {r["sid"] for r in rp.records}:
        assert len({r["priority"] for r in rp.records if r["sid"] == sid}) == 1


def test_summarize_counts_unserved_as_misses():
    cfg = RunConfig(warmup_s=10, duration_s=100)
    rec = lambda arrival, turn, ttft: {"arrival": arrival, "turn": turn, "ttft": ttft}  # noqa: E731
    records = [rec(5, 0, 1.0),  # before the window: ignored
               rec(20, 0, 1.0), rec(30, 1, 2.0), rec(40, 1, 7.0), rec(50, 1, None)]
    s = summarize(records, cfg)
    assert s["requests"] == 4 and s["returning_requests"] == 3 and s["served"] == 3
    assert s["returning_goodput_5s"] == pytest.approx(1 / 100)
    assert s["returning_slo_share_5s"] == pytest.approx(1 / 3)
    assert s["goodput_5s"] == pytest.approx(2 / 100)


# --- live server --------------------------------------------------------------------------------------
@pytest.fixture
def live_traces():
    try:
        httpx.get(f"{BASE}/health", timeout=5).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("no vLLM server on localhost:8000")
    if not SESSIONS.exists():
        pytest.skip(f"{SESSIONS} missing")
    return load_traces([SESSIONS], "all")


@pytest.mark.gpu
def test_replayed_prompts_match_trace_token_for_token(live_traces):
    checked, problems = check(live_traces[:5], BASE)
    assert checked > 20
    assert problems == []


@pytest.mark.gpu
def test_live_replay_one_session(live_traces):
    tr = min(live_traces, key=lambda t: t.session.turns[0].prompt_tokens)
    cfg = RunConfig(max_turns=3, pause_scale=0)

    async def go():
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            model = (await client.get(f"{BASE}/v1/models")).json()["data"][0]["id"]
            rp = Replayer([tr], cfg, client, model)
            await rp.session(tr, f"{tr.session_id}#test", 12345, 0.0)
            return rp.records

    records = asyncio.run(go())
    assert len(records) == 3 and all("error" not in r for r in records)
    assert all(r["ttft"] is not None and 0 < r["ttft"] < r["finish"] - r["arrival"] + 1e-6 for r in records)
    assert all(r["served_output_tokens"] == r["output_tokens"] for r in records)
    deltas = {r["served_prompt_tokens"] - r["prompt_tokens"] for r in records}
    assert len(deltas) == 1 and 0 < deltas.pop() <= 32  # only the salt line differs, the same on every turn
