"""Shared KV placement interface (SPEC.md): written once, used by the simulator (C7) and vLLM (C1).

All times are seconds on one clock (wall clock in vLLM, simulated clock in the simulator), so
policy methods take `now` explicitly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from kvtier.predictor import Prediction


class Action(Enum):
    KEEP = "keep"  # leave KV on GPU; evictable only under pressure, in eviction_order
    OFFLOAD = "offload"  # free GPU blocks now; the CPU copy (LMCache) serves the return
    DISCARD = "discard"  # free GPU blocks now and drop the CPU copy; the return recomputes


class State(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    OFFLOADED = "offloaded"
    DISCARDED = "discarded"


@dataclass
class Session:
    session_id: str
    context_tokens: int = 0
    state: State = State.ACTIVE
    program: str = ""  # tool the agent ran in the pause now in progress
    pause_start: float | None = None
    last_used: float = 0.0  # end of the last request; LRU fallback order
    prediction: Prediction | None = None
    true_pause_s: float | None = None  # oracle only (simulator); never set from live traffic


class KVPolicy(ABC):
    @abstractmethod
    def on_pause(self, s: Session, now: float) -> Action:
        """Called when a session's request finishes and its tool starts."""

    @abstractmethod
    def eviction_order(self, paused: list[Session], now: float) -> list[Session]:
        """Paused sessions still on GPU, first = evict first."""

    def prefetch_at(self, s: Session) -> float | None:
        """Clock time to start reloading an offloaded session, or None for no prefetch."""
        return None

    def on_return(self, s: Session, pause_s: float) -> None:
        """Called when the session's next request arrives; pause_s is the observed pause."""
