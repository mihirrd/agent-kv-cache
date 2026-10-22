"""Raw C5 traces -> replay sessions shared by the simulator (C7) and the replay harness (C6).

A replay session is the trace as the server sees it: per turn, the Qwen-tokenized prompt (as a
chain of 16-token block hashes, like vLLM's prefix cache), the forced output length, the tool pause
that follows, and the tool's program name. Turns past the context cap are dropped.

Build on gpubox:  uv run python -m kvtier.sessions traces/raw/verified traces/sessions/verified.jsonl
"""

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

BLOCK = 16  # vLLM KV block size in tokens


class Tokenizer(Protocol):
    def encode_chat(self, messages: list[dict]) -> list[int]: ...
    def count(self, text: str) -> int: ...


def _h(*parts: int) -> int:
    digest = hashlib.blake2b(b"".join(p.to_bytes(8, "little", signed=True) for p in parts), digest_size=8)
    return int.from_bytes(digest.digest(), "little", signed=True)


def block_hashes(tokens: list[int]) -> list[int]:
    """Hash chain over full 16-token blocks: block i's hash covers all tokens up to its end."""
    out, parent = [], 0
    for i in range(len(tokens) // BLOCK):
        parent = _h(parent, *tokens[i * BLOCK : (i + 1) * BLOCK])
        out.append(parent)
    return out


def salt_hashes(hashes: list[int], keep: int, salt: int) -> list[int]:
    """Make one copy of a session distinct from other copies, except its first `keep` (shared) blocks."""
    return hashes[:keep] + [_h(salt, h) for h in hashes[keep:]]


@dataclass
class Turn:
    prompt_tokens: int
    block_hashes: list[int]
    output_tokens: int
    pause_s: float | None  # tool time after this turn; None on the last turn
    program: str = ""


@dataclass
class ReplaySession:
    session_id: str
    turns: list[Turn] = field(default_factory=list)

    @classmethod
    def from_tokens(cls, session_id, prompts, outputs=None, pauses=None, programs=None):
        """Test helper: a session from raw token lists."""
        n = len(prompts)
        outputs = outputs or [1] * n
        pauses = pauses or [0.0] * (n - 1) + [None]
        programs = programs or [""] * n
        return cls(session_id, [Turn(len(p), block_hashes(p), o, s, g) for p, o, s, g in zip(prompts, outputs, pauses, programs)])


def common_prefix_blocks(sessions: list[ReplaySession]) -> int:
    """Longest block prefix shared by every session's first prompt (system prompt + task template)."""
    firsts = [s.turns[0].block_hashes for s in sessions if s.turns]
    n = 0
    while firsts and all(len(h) > n and h[n] == firsts[0][n] for h in firsts):
        n += 1
    return n


def build_session(path: Path, tok: Tokenizer, context_cap: int) -> ReplaySession:
    lines = [json.loads(line) for line in path.open()]
    records = [r for r in lines if r["type"] == "turn"]
    messages = lines[-1]["messages"] if lines[-1]["type"] == "footer" else []
    turns = []
    for r in records:
        n = r["prompt_messages"]
        tokens = tok.encode_chat(messages[:n])
        if len(tokens) > context_cap:
            break
        reply = r.get("output_text", "") if r.get("format_error") else messages[n]["content"]
        if r.get("format_error"):
            pause = 0.0
        else:
            pause = r["tool_end"] - r["tool_start"] if "tool_end" in r else 0.0
        turns.append(Turn(len(tokens), block_hashes(tokens), tok.count(reply), pause, r.get("program") or ""))
    if turns:
        turns[-1].pause_s = None
    return ReplaySession(path.stem, turns)


def save_sessions(sessions: list[ReplaySession], path: Path, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"meta": meta or {}}) + "\n")
        for s in sessions:
            f.write(json.dumps(asdict(s)) + "\n")


def load_sessions(path: Path) -> tuple[list[ReplaySession], dict]:
    with Path(path).open() as f:
        meta = json.loads(f.readline())["meta"]
        sessions = []
        for line in f:
            d = json.loads(line)
            sessions.append(ReplaySession(d["session_id"], [Turn(**t) for t in d["turns"]]))
    return sessions, meta


class QwenTokenizer:
    def __init__(self, model: str = "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model)

    def encode_chat(self, messages: list[dict]) -> list[int]:
        text = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces", type=Path, help="directory of raw trace .jsonl files")
    ap.add_argument("out", type=Path)
    ap.add_argument("--context-cap", type=int, default=15_500)
    args = ap.parse_args()
    tok = QwenTokenizer()
    sessions = [build_session(p, tok, args.context_cap) for p in sorted(args.traces.glob("*.jsonl"))]
    sessions = [s for s in sessions if s.turns]
    save_sessions(sessions, args.out, meta={"source": str(args.traces), "context_cap": args.context_cap})
    turns = sum(len(s.turns) for s in sessions)
    print(f"{len(sessions)} sessions, {turns} turns, common prefix {common_prefix_blocks(sessions)} blocks -> {args.out}")


if __name__ == "__main__":
    main()
