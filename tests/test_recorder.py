import json

import pytest
from minisweagent.exceptions import FormatError, Submitted

from kvtier.recorder import RecordingAgent, TraceWriter, api_cost, tool_program, usage_fields

TEMPLATES = {"system_template": "sys", "instance_template": "task: {{task}}"}


@pytest.mark.parametrize(
    ("command", "program"),
    [
        ("ls -la", "ls"),
        ("cd /testbed && python -m pytest tests/test_x.py -q", "pytest"),
        ("FOO=1 BAR=2 python3 reproduce.py", "python3"),
        ("/usr/bin/grep -rn foo .", "grep"),
        ("cat <<'EOF' > a.py\nprint(1)\nEOF", "cat"),
        ("cd sub; sed -n 1,20p x.py", "sed"),
        ("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git diff", "echo"),
        ("", ""),
    ],
)
def test_tool_program(command, program):
    assert tool_program(command) == program


def test_usage_fields_and_cost():
    response = {
        "usage": {
            "prompt_tokens": 10_000,
            "completion_tokens": 400,
            "prompt_tokens_details": {"cached_tokens": 8_000},
            "cache_creation_input_tokens": 1_000,
            "completion_tokens_details": {"reasoning_tokens": 150},
        }
    }
    u = usage_fields(response)
    assert u == {
        "api_prompt_tokens": 10_000,
        "api_output_tokens": 400,
        "api_cache_read_tokens": 8_000,
        "api_cache_write_tokens": 1_000,
        "api_reasoning_tokens": 150,
    }
    # 1,000 uncached input + 8,000 cache reads at 0.1x + 1,000 cache writes at 1.25x + 400 output
    expected = (1_000 * 0.10 + 8_000 * 0.01 + 1_000 * 0.125 + 400 * 0.50) / 1e6
    assert api_cost(u) == pytest.approx(expected)
    assert usage_fields(None)["api_prompt_tokens"] == 0


class FakeModel:
    """Scripted assistant replies; reply "BAD" has no action and raises FormatError."""

    def __init__(self, replies):
        self.replies = list(replies)

    def query(self, messages):
        content = self.replies.pop(0)
        response = {"usage": {"prompt_tokens": 100 * len(messages), "completion_tokens": 20}}
        if content == "BAD":
            raise FormatError(
                {
                    "role": "user",
                    "content": "format error",
                    "extra": {"model_response": content, "response": response, "cost": 0.0},
                }
            )
        return {"role": "assistant", "content": content, "extra": {"actions": [{"command": content}], "response": response}}

    def format_message(self, **kwargs):
        return kwargs

    def format_observation_messages(self, message, outputs, template_vars=None):
        return [{"role": "user", "content": f"rc={o['returncode']}"} for o in outputs]

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


class FakeEnv:
    def execute(self, action):
        if action["command"] == "submit":
            raise Submitted({"role": "exit", "content": "patch", "extra": {"exit_status": "Submitted", "submission": "patch"}})
        return {"output": "ok", "returncode": 3 if action["command"].startswith("pytest") else 0}

    def get_template_vars(self, **kwargs):
        return {}

    def serialize(self):
        return {}


def read_trace(path):
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    return lines[0], [r for r in lines if r["type"] == "turn"], lines[-1]


def run_agent(tmp_path, replies, count_tokens=lambda msgs: 10, cap=15_500):
    path = tmp_path / "s.jsonl"
    with TraceWriter(path, header={"instance_id": "repo__1"}) as writer:
        agent = RecordingAgent(
            FakeModel(replies), FakeEnv(), writer=writer, count_tokens=count_tokens, context_cap=cap, **TEMPLATES
        )
        info = agent.run("fix it")
    return info, *read_trace(path)


def test_records_turns_and_footer(tmp_path):
    info, header, turns, footer = run_agent(tmp_path, ["ls", "pytest -x", "BAD", "submit"])
    assert info["exit_status"] == "Submitted"
    assert header["type"] == "header" and header["instance_id"] == "repo__1"
    assert [t["turn"] for t in turns] == [1, 2, 3, 4]
    assert [t["program"] for t in turns] == ["ls", "pytest", None, "submit"]
    assert turns[1]["returncode"] == 3
    assert turns[2]["format_error"] and turns[2]["output_text"] == "BAD"
    assert turns[3]["submitted"]
    for t in turns:
        assert t["request_start"] <= t["request_end"]
        if t["program"]:
            assert t["request_end"] <= t["tool_start"] <= t["tool_end"]
    # Each turn's prompt is a prefix of the footer's message list.
    msgs = footer["messages"]
    assert footer["type"] == "footer" and footer["exit_status"] == "Submitted"
    assert [m["role"] for m in msgs[:2]] == ["system", "user"]
    assert turns[0]["prompt_messages"] == 2
    assert msgs[2] == {"role": "assistant", "content": "ls"}
    assert turns[1]["prompt_messages"] == 4  # + assistant "ls" + observation
    assert turns[3]["prompt_messages"] == turns[2]["prompt_messages"] + 1  # format error adds one user message
    assert all("extra" not in m for m in msgs)
    assert footer["cost"] == pytest.approx(sum(t["cost"] for t in turns))


def test_context_cap_stops_before_request(tmp_path):
    info, _, turns, footer = run_agent(
        tmp_path, ["ls", "ls", "ls"], count_tokens=lambda msgs: 1000 * len(msgs), cap=4_500
    )
    # Prompts have 2 then 4 messages; the third (6 messages = 6,000 tokens) exceeds the cap.
    assert info["exit_status"] == "ContextCap"
    assert len(turns) == 2
    assert footer["exit_status"] == "ContextCap"
    assert turns[1]["qwen_prompt_tokens"] == 4000


@pytest.mark.gpu  # needs the Qwen tokenizer in gpubox's HF cache
def test_qwen_token_counter_counts_tokens():
    from kvtier.recorder import qwen_token_counter

    count = qwen_token_counter("Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    short = count([{"role": "user", "content": "hi"}])
    long = count([{"role": "user", "content": "hello world " * 500}])
    assert 5 < short < 50
    assert long - short > 900
