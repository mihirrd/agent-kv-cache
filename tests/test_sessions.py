import json

import pytest

from kvtier.sessions import (
    BLOCK,
    ReplaySession,
    block_hashes,
    build_session,
    common_prefix_blocks,
    load_sessions,
    save_sessions,
    salt_hashes,
)


class FakeTokenizer:
    """One token per character of the rendered chat; deterministic and easy to count."""

    def encode_chat(self, messages):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return [ord(c) for c in text]

    def count(self, text):
        return len(text)


def write_trace(path, turns, messages):
    lines = [{"type": "header", "instance_id": path.stem}, {"type": "env_ready", "seconds": 0.1}]
    lines += [{"type": "turn", **t} for t in turns]
    lines.append({"type": "footer", "exit_status": "Submitted", "messages": messages})
    path.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return path


def test_block_hashes_chain_and_drop_partial_block():
    toks = list(range(40))
    h = block_hashes(toks)
    assert len(h) == 40 // BLOCK
    assert block_hashes(toks[:32]) == h  # same full blocks, same hashes
    assert block_hashes([99] + toks[1:])[0] != h[0]
    # chained: a change in block 0 changes every later hash too
    assert block_hashes([99] + toks[1:])[1] != h[1]


def test_build_session(tmp_path):
    sys_msg = {"role": "system", "content": "S" * 40}
    task = {"role": "user", "content": "T" * 30}
    a1 = {"role": "assistant", "content": "x" * 20}
    o1 = {"role": "user", "content": "o" * 50}
    err = {"role": "user", "content": "format error"}
    a3 = {"role": "assistant", "content": "z" * 9}
    o3 = {"role": "user", "content": "done"}
    turns = [
        {"turn": 1, "prompt_messages": 2, "program": "grep", "tool_start": 10.0, "tool_end": 12.5},
        {"turn": 2, "prompt_messages": 4, "program": None, "format_error": True, "output_text": "bad reply"},
        {"turn": 3, "prompt_messages": 5, "program": "echo", "tool_start": 20.0, "tool_end": 20.1, "submitted": True},
    ]
    path = write_trace(tmp_path / "repo__1.jsonl", turns, [sys_msg, task, a1, o1, err, a3, o3])
    s = build_session(path, FakeTokenizer(), context_cap=10_000)
    tok = FakeTokenizer()
    assert s.session_id == "repo__1" and len(s.turns) == 3
    t1, t2, t3 = s.turns
    assert t1.prompt_tokens == len(tok.encode_chat([sys_msg, task]))
    assert t1.output_tokens == 20 and t2.output_tokens == len("bad reply") and t3.output_tokens == 9
    assert t1.pause_s == pytest.approx(2.5)
    assert t2.pause_s == 0.0  # format error: the agent re-asks immediately
    assert t3.pause_s is None  # last turn
    assert t1.program == "grep" and t2.program == "" and t3.program == "echo"
    # each turn's prompt starts with the previous turn's prompt (an assistant reply follows the
    # generation prompt) ...
    assert t2.block_hashes[: len(t1.block_hashes)] == t1.block_hashes
    # ... except after a format error, where a user message follows it: only the block holding
    # the generation prompt can differ
    assert t3.block_hashes[: len(t2.block_hashes) - 1] == t2.block_hashes[:-1]


def test_context_cap_drops_later_turns(tmp_path):
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "T" * 50}]
    msgs += [{"role": "assistant", "content": "a" * 10}, {"role": "user", "content": "o" * 100}]
    turns = [
        {"turn": 1, "prompt_messages": 2, "program": "ls", "tool_start": 0.0, "tool_end": 1.0},
        {"turn": 2, "prompt_messages": 4, "program": "ls", "tool_start": 2.0, "tool_end": 3.0},
    ]
    path = write_trace(tmp_path / "x.jsonl", turns, msgs)
    s = build_session(path, FakeTokenizer(), context_cap=120)
    assert len(s.turns) == 1 and s.turns[0].pause_s is None  # turn 1 becomes the last turn


def test_common_prefix_and_salting():
    shared = list(range(48))
    a = ReplaySession.from_tokens("a", [shared + [1] * 32])
    b = ReplaySession.from_tokens("b", [shared + [2] * 32])
    n = common_prefix_blocks([a, b])
    assert n == 3
    h = a.turns[0].block_hashes
    s1, s2 = salt_hashes(h, n, salt=1), salt_hashes(h, n, salt=2)
    assert s1[:n] == s2[:n] == h[:n]  # shared prefix still hits across copies
    assert s1[n:] != s2[n:] and s1[n:] != h[n:]
    assert salt_hashes(h, n, salt=1) == s1  # deterministic: turns of one copy stay consistent


def test_save_and_load_roundtrip(tmp_path):
    a = ReplaySession.from_tokens("a", [list(range(40)), list(range(70))], outputs=[5, 6], pauses=[1.5, None],
                                  programs=["pytest", "echo"])
    save_sessions([a], tmp_path / "s.jsonl", meta={"set": "test"})
    (b,), meta = load_sessions(tmp_path / "s.jsonl")
    assert b == a and meta["set"] == "test"
