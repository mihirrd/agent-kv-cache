"""C2 pause predictor: per-key ring buffers of observed tool pauses, with back-off.

Keys, most specific first: (session, program), (program), (tool category), global. A prediction
comes from the first key with at least `min_samples` pauses and is confident when the p90/p10
spread is at most `max_spread`. Pure Python so the simulator and vLLM share it.

The history can be saved and loaded (`save` / `load`) so a restarted server keeps what it learned
about each tool. Only the shared keys are saved, newest pauses last at 0.1 s resolution, capped at
`MAX_SAVED_PROGRAMS` programs, least recently seen dropped first. Per-session keys stay in memory:
they describe one agent's run and end with it.
"""

import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

MAX_SAVED_PROGRAMS = 256  # one-off names (script paths) must not grow the file without bound

_CATEGORIES = {
    "test": {"pytest", "py.test", "tox", "nox", "nosetests", "runtests.py", "unittest", "ctest", "jest", "mocha", "rspec"},
    "build": {"cargo", "go", "mvn", "gradle", "gradlew", "make", "cmake", "ninja", "npm", "yarn", "pnpm", "javac",
              "gcc", "g++", "clang", "rustc", "bundle", "composer", "ant", "sbt"},
    "install": {"pip", "pip3", "conda", "apt-get", "apt", "uv", "poetry"},
    "inspect": {"grep", "rg", "find", "ls", "cat", "head", "tail", "sed", "awk", "wc", "git", "echo", "diff", "nl",
                "tree", "which", "pwd", "file", "stat"},
    "run": {"python", "python3", "node", "ruby", "php", "java", "bash", "sh", "perl"},
}
_CATEGORY_OF = {program: category for category, programs in _CATEGORIES.items() for program in programs}


def tool_category(program: str) -> str:
    """Coarse tool class used as the third back-off key."""
    return _CATEGORY_OF.get(program, "other")


def percentile(sorted_xs: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already sorted, non-empty list."""
    pos = q * (len(sorted_xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


@dataclass(frozen=True)
class Prediction:
    p10: float
    p50: float
    p90: float
    n: int
    confident: bool
    key: tuple


class PausePredictor:
    def __init__(self, window: int = 50, min_samples: int = 5, max_spread: float = 20.0, min_pause: float = 1e-3):
        self.window = window
        self.min_samples = min_samples
        self.max_spread = max_spread
        self.min_pause = min_pause  # floor for p10 in the spread test, so all-zero buffers are confident
        self._buffers: dict[tuple, deque[float]] = {}

    @staticmethod
    def keys(session_id: str, program: str) -> list[tuple]:
        return [("session", session_id, program), ("program", program), ("tool", tool_category(program)), ("global",)]

    def observe(self, session_id: str, program: str, pause_s: float) -> None:
        for key in self.keys(session_id, program):
            # Re-inserted so the dict's order is least recently seen first (used when saving).
            buf = self._buffers.pop(key, None) or deque(maxlen=self.window)
            buf.append(pause_s)
            self._buffers[key] = buf

    def to_dict(self) -> dict:
        """The saved history: per-program, per-category and global pauses, oldest first."""
        out: dict = {"version": 1, "programs": {}, "tools": {}, "global": []}
        programs = [k for k in self._buffers if k[0] == "program"][-MAX_SAVED_PROGRAMS:]
        for key in programs + [k for k in self._buffers if k[0] in ("tool", "global")]:
            pauses = [round(x, 1) for x in self._buffers[key]]
            if key[0] == "global":
                out["global"] = pauses
            else:
                out[key[0] + "s"][key[1]] = pauses
        return out

    def load_dict(self, d: dict) -> None:
        """Seed the shared keys from a saved history (the last `window` pauses of each)."""
        entries = [(("program", p), xs) for p, xs in d.get("programs", {}).items()]
        entries += [(("tool", t), xs) for t, xs in d.get("tools", {}).items()]
        entries.append((("global",), d.get("global", [])))
        for key, xs in entries:
            if xs:
                self._buffers[key] = deque(xs, maxlen=self.window)

    def save(self, path: str | Path) -> None:
        """Write the history atomically, so a crash mid-write never leaves a broken file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), separators=(",", ":")))
        os.replace(tmp, path)

    def load(self, path: str | Path) -> int:
        """Load a saved history if the file exists; returns the number of programs loaded."""
        path = Path(path)
        if not path.exists():
            return 0
        d = json.loads(path.read_text())
        self.load_dict(d)
        return len(d.get("programs", {}))

    def remaining(self, session_id: str, program: str, elapsed: float) -> float | None:
        """Expected time left in a pause that has already lasted `elapsed` seconds: the median of
        (x - elapsed) over past pauses x longer than `elapsed`, from the first back-off key with
        enough samples. Infinite when no past pause ran this long; None with no history at all."""
        for key in self.keys(session_id, program):
            buf = self._buffers.get(key)
            if buf is not None and len(buf) >= self.min_samples:
                left = sorted(x - elapsed for x in buf if x > elapsed)
                return percentile(left, 0.5) if left else float("inf")
        return None

    def predict(self, session_id: str, program: str) -> Prediction | None:
        for key in self.keys(session_id, program):
            buf = self._buffers.get(key)
            if buf is not None and len(buf) >= self.min_samples:
                xs = sorted(buf)
                p10, p50, p90 = (percentile(xs, q) for q in (0.1, 0.5, 0.9))
                confident = p90 / max(p10, self.min_pause) <= self.max_spread
                return Prediction(p10, p50, p90, len(xs), confident, key)
        return None
