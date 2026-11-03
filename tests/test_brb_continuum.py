"""Continuum baseline (fork module vllm/v1/core/brb_continuum.py): pin rule, queue order, and pins
holding blocks in vLLM's real scheduler. CPU only; no GPU or server needed."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm.v1.core.brb_continuum import (  # noqa: E402
    PIN_S,
    Continuum,
    ContinuumRequestQueue,
)
from vllm.v1.request import RequestStatus  # noqa: E402


def fake(rid: str, sid: str, program: str | None = "ls", pause_s: float | None = 1.0):
    extra = {"session_id": sid}
    if program:
        extra["program"] = program
    if pause_s is not None:
        extra["pause_s"] = pause_s
    return SimpleNamespace(request_id=rid, arrival_time=0.0, sampling_params=SimpleNamespace(extra_args=extra))


def turn(c: Continuum, sid: str, t_in: float, t_out: float, **kw):
    r = fake(f"{sid}@{t_in}", sid, **kw)
    c.on_arrival(r, t_in)
    return r, c.on_finish(r, t_out)


def test_unseen_tool_is_pinned_for_two_seconds():
    c = Continuum()
    r, (release, pin) = turn(c, "a", 0.0, 1.0)
    assert pin and release == [] and c.pinned["a"] == (r, 1.0 + PIN_S)


def test_tool_averaging_over_two_seconds_is_not_pinned():
    c = Continuum()
    turn(c, "a", 0.0, 1.0, program="pytest")
    _, (release, pin) = turn(c, "a", 6.0, 7.0, program="pytest")  # pytest took 5 s
    assert not pin
    assert [r.request_id for r in release] == ["a@0.0"]  # the earlier pin goes when the next finishes
    _, (_, pin) = turn(c, "b", 7.0, 8.0, program="ls")  # other tools unaffected
    assert pin


def test_last_step_and_no_tool_are_not_pinned():
    c = Continuum()
    assert turn(c, "a", 0.0, 1.0, pause_s=None)[1] == ([], False)
    assert turn(c, "b", 0.0, 1.0, program=None)[1] == ([], False)


def test_pins_expire_unless_the_session_is_waiting():
    c = Continuum()
    turn(c, "a", 0.0, 1.0)
    turn(c, "b", 0.0, 1.0)
    assert c.expired_pins(set(), 2.0) == []
    out = c.expired_pins({"b"}, 1.0 + PIN_S)
    assert [r.request_id for r in out] == ["a@0.0"] and list(c.pinned) == ["b"]


def test_release_latest_takes_the_pin_that_ends_last():
    c = Continuum()
    turn(c, "a", 0.0, 1.0)
    turn(c, "b", 0.0, 1.5)
    assert c.release_latest().request_id == "b@0.0" and list(c.pinned) == ["a"]


def test_waiting_order_is_pinned_sessions_then_session_arrival():
    c = Continuum()
    q = ContinuumRequestQueue(c)
    for sid, t in (("old", 0.0), ("mid", 1.0), ("new", 2.0)):
        c.on_arrival(fake(sid, sid), t)
    c.pinned["new"] = (fake("x", "new"), 99.0)
    for sid in ("mid", "old", "new"):
        q.add_request(fake(f"{sid}-req", sid))
    assert [q.pop_request().request_id for _ in range(3)] == ["new-req", "old-req", "mid-req"]


def test_preempts_latest_session_and_spares_last_steps():
    c = Continuum()
    c.on_arrival(fake("o", "old"), 0.0)
    c.on_arrival(fake("n", "new"), 5.0)
    old, new_last = fake("o2", "old"), fake("n2", "new", pause_s=None)
    assert max([old, new_last], key=c.preempt_key) is old


def _scheduler():
    sys.path.insert(0, str(Path(__file__).parents[1] / "third_party/vllm"))
    from tests.v1.core.utils import create_requests, create_scheduler

    s = create_scheduler(enable_prefix_caching=True, num_blocks=64)
    s.continuum = Continuum()
    s.waiting = ContinuumRequestQueue(s.continuum)
    (req,) = create_requests(1, num_tokens=64, req_ids=["r0"])
    req.sampling_params.extra_args = {"session_id": "a", "program": "ls", "pause_s": 1.0}
    return s, req


def test_pinned_request_keeps_its_blocks_until_the_pin_expires():
    s, req = _scheduler()
    pool = s.kv_cache_manager.block_pool
    free0 = pool.get_num_free_blocks()
    s.add_request(req)
    s.schedule()
    s.finish_requests("r0", RequestStatus.FINISHED_ABORTED)
    assert "a" in s.continuum.pinned and "r0" not in s.requests
    assert pool.get_num_free_blocks() < free0  # held: LRU cannot take them
    s.continuum.pinned["a"] = (req, 0.0)  # expire it
    s.schedule()
    assert not s.continuum.pinned and pool.get_num_free_blocks() == free0
