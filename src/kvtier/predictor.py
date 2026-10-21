"""C2 pause predictor: per-key ring buffers of observed tool pauses, with back-off.

Keys, most specific first: (session, program), (program), (tool category), global. A prediction
comes from the first key with at least `min_samples` pauses and is confident when the p90/p10
spread is at most `max_spread`. Pure Python so the simulator and vLLM share it.
"""

from collections import deque
from dataclasses import dataclass

_CATEGORIES = {
    "test": {"pytest", "py.test", "tox", "nox", "nosetests", "runtests.py", "unittest", "ctest", "jest", "mocha", "rspec"},
    "build": {"cargo", "go", "mvn", "gradle", "gradlew", "make", "cmake", "ninja", "npm", "yarn", "pnpm", "javac",
              "gcc", "g++", "clang", "rustc", "bundle", "composer"},
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
            self._buffers.setdefault(key, deque(maxlen=self.window)).append(pause_s)

    def predict(self, session_id: str, program: str) -> Prediction | None:
        for key in self.keys(session_id, program):
            buf = self._buffers.get(key)
            if buf is not None and len(buf) >= self.min_samples:
                xs = sorted(buf)
                p10, p50, p90 = (percentile(xs, q) for q in (0.1, 0.5, 0.9))
                confident = p90 / max(p10, self.min_pause) <= self.max_spread
                return Prediction(p10, p50, p90, len(xs), confident, key)
        return None
