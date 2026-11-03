import json

from brb.plan import keeps_up, live_kv, main, pick_load

OK = {"errors": 0, "returning_slo_share_2s": 1.0, "waiting_avg": 0.0}
SLOW = {"errors": 0, "returning_slo_share_2s": 0.94, "waiting_avg": 0.1}


def test_keeps_up():
    assert keeps_up(OK)
    assert not keeps_up(SLOW)
    assert not keeps_up({**OK, "waiting_avg": 3.0})
    assert not keeps_up({**OK, "errors": 2})


def test_pick_load_stops_at_first_failure():
    assert pick_load({2: OK, 4: OK, 8: SLOW, 12: OK}) == 4  # 12 passing is noise past the knee
    assert pick_load({2: SLOW, 4: OK}) is None
    assert pick_load({}) is None


def test_pick_load_stops_when_turns_stretch():
    # on time and no queue, but agents spend far less time in tools than at the lowest load: overloaded
    loads = {8: {**OK, "pause_share": 0.57}, 16: {**OK, "pause_share": 0.45}, 60: {**OK, "pause_share": 0.2}}
    assert pick_load(loads) == 16


def test_live_kv_splits_running_and_paused():
    # one session: request 0-2 s (100+10 tokens), pause 2-6 s, request 6-8 s (200+20), then it ends
    recs = [{"sid": "a", "turn": 0, "arrival": 0.0, "finish": 2.0, "prompt_tokens": 100, "output_tokens": 10},
            {"sid": "a", "turn": 1, "arrival": 6.0, "finish": 8.0, "prompt_tokens": 200, "output_tokens": 20}]
    running, paused = live_kv(recs, 0.0, 10.0)
    assert running == (2 * 110 + 2 * 220) / 10
    assert paused == 4 * 110 / 10


def test_live_kv_unfinished_request_runs_to_the_end():
    recs = [{"sid": "a", "turn": 0, "arrival": 5.0, "finish": None, "prompt_tokens": 100, "output_tokens": 10}]
    assert live_kv(recs, 0.0, 10.0) == (50.0, 0.0)


def test_cli(tmp_path, capsys):
    for load, s in ((4, OK), (8, SLOW)):
        d = tmp_path / f"load{load}" / "full" / "seed0"
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(s))
    main(["load", str(tmp_path), "4", "8", "12"])
    assert capsys.readouterr().out.strip() == "4"
    run = tmp_path / "load4" / "full" / "seed0"
    (run / "config.json").write_text(json.dumps({"warmup_s": 0, "duration_s": 10}))
    (run / "requests.jsonl").write_text(json.dumps(
        {"sid": "a", "turn": 0, "arrival": 0.0, "finish": 10.0, "prompt_tokens": 990, "output_tokens": 10}) + "\n")
    main(["pools", str(run), "1.5", "2"])
    assert capsys.readouterr().out.strip() == "2000 2000"  # 1,500 and 2,000 rounded to 1K
