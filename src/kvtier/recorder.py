"""C5 trace recorder: run mini-SWE-agent on SWE-bench tasks and log one JSONL trace per session.

Each trace has a header line, one line per model call ("turn") and a footer with the full message
list. Turn i's prompt is footer["messages"][:prompt_messages], so replay can rebuild every prompt
exactly. Timestamps are wall-clock seconds; pauses are tool_end - tool_start of each turn.

Run on gpubox (needs Docker and ANTHROPIC_API_KEY):
    uv run python -m kvtier.recorder --n 10 --out traces/raw/pilot
"""

import argparse
import concurrent.futures
import json
import re
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, LimitsExceeded, Submitted

# Claude Haiku 5.5, $ per 1M tokens (prompts up to 100K). Cache multipliers follow the standard
# Anthropic pricing (reads 0.1x, 5-minute writes 1.25x).
PRICE_INPUT, PRICE_OUTPUT = 0.10, 0.50
PRICE_CACHE_READ, PRICE_CACHE_WRITE = 0.1 * PRICE_INPUT, 1.25 * PRICE_INPUT

_ENV_ASSIGN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
_SEGMENT_SPLIT = re.compile(r"&&|\|\||;|\|")


def tool_program(command: str) -> str:
    """Program a command runs, as the predictor's key: skips cd/export and VAR=x prefixes."""
    first_line = command.strip().splitlines()[0] if command.strip() else ""
    for segment in _SEGMENT_SPLIT.split(first_line):
        words = [w for w in segment.split() if not _ENV_ASSIGN.match(w)]
        if not words or words[0] in ("cd", "export"):
            continue
        program = words[0].rsplit("/", 1)[-1]
        if program.startswith("python") and len(words) > 2 and words[1] == "-m":
            return words[2]
        return program
    return ""


def usage_fields(response: dict | None) -> dict:
    """Token usage from a LiteLLM response dump (Anthropic provider)."""
    usage = (response or {}).get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    output_details = usage.get("completion_tokens_details") or {}
    return {
        "api_prompt_tokens": usage.get("prompt_tokens") or 0,
        "api_output_tokens": usage.get("completion_tokens") or 0,
        "api_cache_read_tokens": prompt_details.get("cached_tokens") or 0,
        "api_cache_write_tokens": usage.get("cache_creation_input_tokens") or 0,
        "api_reasoning_tokens": output_details.get("reasoning_tokens") or 0,
    }


def api_cost(u: dict) -> float:
    """Dollar cost of one call; api_prompt_tokens includes cache reads and writes."""
    uncached = u["api_prompt_tokens"] - u["api_cache_read_tokens"] - u["api_cache_write_tokens"]
    return (
        uncached * PRICE_INPUT
        + u["api_cache_read_tokens"] * PRICE_CACHE_READ
        + u["api_cache_write_tokens"] * PRICE_CACHE_WRITE
        + u["api_output_tokens"] * PRICE_OUTPUT
    ) / 1e6


def api_messages(messages: list[dict]) -> list[dict]:
    """Role and content only, as replay sends them; drops mini-SWE-agent's exit marker."""
    return [{"role": m["role"], "content": m.get("content") or ""} for m in messages if m["role"] != "exit"]


class TraceWriter:
    """Appends JSON lines to one session file, flushing each line so a crash keeps the turns so far."""

    def __init__(self, path: Path, header: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._f = path.open("w")
        self.write({"type": "header", "started": time.time(), **header})

    def write(self, record: dict) -> None:
        self._f.write(json.dumps(record) + "\n")
        self._f.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()


class RecordingAgent(DefaultAgent):
    """DefaultAgent that writes a trace line per model call and stops at a Qwen-token context cap."""

    def __init__(self, *args, writer: TraceWriter, count_tokens: Callable[[list[dict]], int], context_cap: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = writer
        self.count_tokens = count_tokens
        self.context_cap = context_cap
        self.total_cost = 0.0
        self._turn: dict = {}
        self._returncodes: list[int] = []

    def run(self, task: str = "", **kwargs) -> dict:
        info = {}
        try:
            info = super().run(task, **kwargs)
            return info
        finally:
            self.writer.write(
                {
                    "type": "footer",
                    "ended": time.time(),
                    "exit_status": info.get("exit_status") or (self.messages[-1].get("extra", {}).get("exit_status") if self.messages else ""),
                    "n_turns": self.n_calls,
                    "cost": self.total_cost,
                    "messages": api_messages(self.messages),
                }
            )

    def query(self) -> dict:
        prompt = api_messages(self.messages)
        n_tokens = self.count_tokens(prompt)
        if n_tokens > self.context_cap:
            raise LimitsExceeded(
                {"role": "exit", "content": "ContextCap", "extra": {"exit_status": "ContextCap", "submission": ""}}
            )
        self._turn = {"type": "turn", "turn": self.n_calls + 1, "prompt_messages": len(prompt), "qwen_prompt_tokens": n_tokens}
        self._turn["request_start"] = time.time()
        try:
            message = super().query()
        except FormatError as e:
            extra = e.messages[0].get("extra", {})
            self._finish_request(extra.get("response"))
            self._turn |= {"format_error": True, "output_text": extra.get("model_response", ""), "program": None}
            self._emit()
            raise
        self._finish_request(message.get("extra", {}).get("response"))
        return message

    def _finish_request(self, response) -> None:
        self._turn["request_end"] = time.time()
        if not isinstance(response, dict):
            response = None
        u = usage_fields(response)
        cost = api_cost(u)
        self.total_cost += cost
        self.cost += cost  # makes cost_limit use our prices; LiteLLM may not price this model
        self._turn |= u | {"cost": cost}

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        command = actions[0]["command"] if actions else ""
        self._turn |= {"command": command, "program": tool_program(command)}
        self._turn["tool_start"] = time.time()
        outputs = []
        try:
            for action in actions:
                outputs.append(self.env.execute(action))
        except Submitted:
            self._turn |= {"tool_end": time.time(), "submitted": True}
            self._emit()
            raise
        self._turn |= {"tool_end": time.time(), "returncode": outputs[-1]["returncode"] if outputs else None}
        self._emit()
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))

    def _emit(self) -> None:
        self.writer.write(self._turn)
        self._turn = {}


def qwen_token_counter(model_name: str) -> Callable[[list[dict]], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lock = threading.Lock()

    def count(messages: list[dict]) -> int:
        with lock:
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count


def record_instance(instance: dict, config: dict, out_dir: Path, count_tokens, context_cap: int) -> dict:
    from minisweagent.models import get_model
    from minisweagent.run.benchmarks.swebench import get_sb_environment

    iid = instance["instance_id"]
    header = {"instance_id": iid, "repo": instance.get("repo"), "model": config["model"]["model_name"], "context_cap": context_cap}
    with TraceWriter(out_dir / f"{iid}.jsonl", header=header) as writer:
        env_start = time.time()
        env = get_sb_environment(config, instance)
        writer.write({"type": "env_ready", "seconds": time.time() - env_start})
        agent = RecordingAgent(
            get_model(config=dict(config["model"])), env,
            writer=writer, count_tokens=count_tokens, context_cap=context_cap, **config["agent"],
        )
        try:
            info = agent.run(instance["problem_statement"])
        finally:
            env.cleanup()
    return {"instance_id": iid, "exit_status": info.get("exit_status"), "turns": agent.n_calls, "cost": agent.total_cost}


def main() -> None:
    from datasets import load_dataset
    from minisweagent.config import builtin_config_dir, get_config_from_spec
    from minisweagent.run.benchmarks.swebench import filter_instances
    from minisweagent.utils.serialize import recursive_merge

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=10, help="number of tasks")
    ap.add_argument("--offset", type=int, default=0, help="skip this many tasks of the shuffled list")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="anthropic/claude-haiku-5-5")
    ap.add_argument("--context-cap", type=int, default=15_500, help="stop before a prompt exceeds this many Qwen tokens")
    ap.add_argument("--task-cost-limit", type=float, default=0.50)
    ap.add_argument("--step-limit", type=int, default=80)
    ap.add_argument("--qwen-model", default="Qwen/Qwen2.5-Coder-7B-Instruct-AWQ")
    args = ap.parse_args()

    config = recursive_merge(
        get_config_from_spec(builtin_config_dir / "benchmarks" / "swebench_backticks.yaml"),
        {
            "model": {
                "model_class": "litellm_textbased",
                "model_name": args.model,
                "cost_tracking": "ignore_errors",
                "set_cache_control": "default_end",
            },
            "agent": {"step_limit": args.step_limit, "cost_limit": args.task_cost_limit},
            "environment": {"pull_timeout": 1800},
        },
    )

    instances = list(load_dataset("princeton-nlp/SWE-Bench_Verified", split="test"))
    instances = filter_instances(instances, filter_spec="", shuffle=True)[args.offset : args.offset + args.n]
    done = {p.stem for p in args.out.glob("*.jsonl")} if args.out.exists() else set()
    todo = [i for i in instances if i["instance_id"] not in done]
    print(f"{len(todo)} tasks to record ({len(instances) - len(todo)} already done) -> {args.out}", flush=True)

    count_tokens = qwen_token_counter(args.qwen_model)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(record_instance, i, config, args.out, count_tokens, args.context_cap): i["instance_id"] for i in todo}
        for future in concurrent.futures.as_completed(futures):
            try:
                r = future.result()
            except Exception:
                r = {"instance_id": futures[future], "exit_status": "error", "turns": 0, "cost": 0.0}
                traceback.print_exc()
            results.append(r)
            total = sum(x["cost"] for x in results)
            print(f"[{len(results)}/{len(todo)}] {r['instance_id']}: {r['exit_status']}, {r['turns']} turns, ${r['cost']:.3f} (total ${total:.3f})", flush=True)


if __name__ == "__main__":
    main()
