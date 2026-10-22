"""Discrete-event loop: Poisson session arrivals, recorded turns and pauses, server steps, prefetch."""

import heapq
import itertools
import math
import random
from dataclasses import dataclass, field

from kvtier.cost import CostModel, PredictivePolicy
from kvtier.policy import Action, KVPolicy, Session, State
from kvtier.sessions import BLOCK, ReplaySession, salt_hashes
from kvtier.sim.memory import CpuTier, GpuPool
from kvtier.sim.server import Request, Server
from kvtier.sim.timing import StepModel

# Event priorities at equal times: finish work before admitting new arrivals.
STEP_END, TURN, PREFETCH, ARRIVAL, SAMPLE = 0, 1, 2, 3, 4
SAMPLE_EVERY_S = 1.0


@dataclass
class SimConfig:
    load_per_min: float  # session arrivals per minute
    cap_tokens: int  # GPU KV capacity
    cpu_tier: bool = True
    cpu_cap_tokens: int = 348_000  # 20 GB at 57,344 bytes/token
    prefetch: bool = True
    warmup_s: float = 120.0
    duration_s: float = 600.0
    drain_s: float = 120.0  # keep running after the window so in-window requests can finish
    seed: int = 0
    program_fcfs: bool = False  # Continuum's queue order: pinned first, then by session arrival


@dataclass(eq=False)
class SimSession:
    replay: ReplaySession
    state: Session
    hashes: list  # salted block hashes per turn
    turn: int = 0
    prefetch_token: int = 0  # bumped to cancel a scheduled prefetch
    warmed: bool = False  # a warm-up completed during the current pause
    pause_rec: dict | None = None  # record of the current pause
    warmup_rec: dict | None = None  # record of the latest warm-up


@dataclass
class Records:
    requests: list = field(default_factory=list)
    pauses: list = field(default_factory=list)
    warmups: list = field(default_factory=list)
    arrivals: list = field(default_factory=list)  # (time, turn) of every agent request submitted
    memory: list = field(default_factory=list)  # (time, blocks in use, blocks cached by paused sessions, blocks used)
    steps: int = 0


class Simulator:
    def __init__(self, sessions: list[ReplaySession], policy: KVPolicy, cost: CostModel, cfg: SimConfig,
                 shared_blocks: int):
        self.sessions = sessions
        self.policy = policy
        self.cost = cost
        self.cfg = cfg
        self.shared_blocks = shared_blocks
        self.gpu = GpuPool(cfg.cap_tokens // BLOCK)
        self.cpu = CpuTier(cfg.cpu_cap_tokens) if cfg.cpu_tier else None
        queue_key = (lambda r: (not r.pinned, r.program_arrival, r.arrival)) if cfg.program_fcfs else None
        self.server = Server(self.gpu, self.cpu, StepModel(cost), shared_blocks, self.make_room, queue_key)
        self.rng = random.Random(cfg.seed)
        self.events: list = []
        self.seq = itertools.count()
        self.live: dict[str, SimSession] = {}  # every session copy that has arrived
        self.step_pending = False
        self.copies = itertools.count()
        self.now = 0.0
        self.rec = Records()

    # --- events ---------------------------------------------------------------------------------
    def push(self, t: float, prio: int, kind: str, payload=None) -> None:
        heapq.heappush(self.events, (t, prio, next(self.seq), kind, payload))

    def run(self, fixed_arrivals: list[tuple[float, ReplaySession]] | None = None) -> Records:
        """Poisson arrivals at cfg.load_per_min, or exactly `fixed_arrivals` (time, session) for tests."""
        end = self.cfg.warmup_s + self.cfg.duration_s + self.cfg.drain_s
        if fixed_arrivals is None:
            self.push(self.rng.expovariate(self.cfg.load_per_min / 60), ARRIVAL, "arrival")
        for t, rs in fixed_arrivals or ():
            self.push(t, ARRIVAL, "arrival", rs)
        self.push(0.0, SAMPLE, "sample")
        while self.events:
            t, _, _, kind, payload = heapq.heappop(self.events)
            if t > end:
                break
            self.now = t
            getattr(self, f"on_{kind}")(payload)
            self.kick()
        return self.rec

    def kick(self) -> None:
        if self.step_pending:
            return
        plan = self.server.plan(self.now)
        if plan is not None:
            self.step_pending = True
            self.rec.steps += 1
            self.push(self.now + plan.duration, STEP_END, "step_end", plan)

    def on_arrival(self, fixed: ReplaySession | None) -> None:
        rs = fixed or self.rng.choice(self.sessions)
        sid = f"{rs.session_id}#{next(self.copies)}"
        salt = self.rng.getrandbits(62)
        hashes = [salt_hashes(t.block_hashes, self.shared_blocks, salt) for t in rs.turns]
        ss = SimSession(rs, Session(sid, program_arrival=self.now), hashes)
        self.live[sid] = ss
        self.submit_turn(ss)
        if fixed is None:
            self.push(self.now + self.rng.expovariate(self.cfg.load_per_min / 60), ARRIVAL, "arrival")

    def on_turn(self, ss: SimSession) -> None:
        s = ss.state
        ss.pause_rec["kv_at_return"] = self.locate(ss)
        self.policy.on_return(s, self.now - s.pause_start)
        ss.prefetch_token += 1
        ss.turn += 1
        self.submit_turn(ss)

    def on_sample(self, _) -> None:
        paused = sum(self.gpu.cached_of(sid) for sid, ss in self.live.items() if ss.state.state is State.PAUSED)
        self.rec.memory.append((self.now, self.gpu.in_use(), paused, self.gpu.used))
        self.push(self.now + SAMPLE_EVERY_S, SAMPLE, "sample")

    def on_prefetch(self, payload) -> None:
        ss, token = payload
        if token != ss.prefetch_token or ss.state.state is not State.OFFLOADED:
            return
        t = ss.replay.turns[ss.turn]
        r = Request(ss.state.session_id, ss.turn, ss.hashes[ss.turn], t.prompt_tokens, 1, self.now, warmup=True)
        self.server.submit(r)
        ss.warmup_rec = {"sid": r.sid, "turn": r.turn, "issued": self.now}
        self.rec.warmups.append(ss.warmup_rec)

    def on_step_end(self, plan) -> None:
        self.step_pending = False
        for r in self.server.apply(plan, self.now):
            if r.warmup:
                self.finish_warmup(r)
            else:
                self.finish_turn(r)

    # --- session lifecycle ------------------------------------------------------------------------
    def submit_turn(self, ss: SimSession) -> None:
        t = ss.replay.turns[ss.turn]
        prev = len(ss.hashes[ss.turn - 1]) if ss.turn else 0
        s = ss.state
        pinned = ss.turn > 0 and self.now < s.pinned_until
        s.state = State.ACTIVE
        s.pinned_until = 0.0
        self.rec.arrivals.append((self.now, ss.turn))
        self.server.submit(Request(s.session_id, ss.turn, ss.hashes[ss.turn], t.prompt_tokens, t.output_tokens,
                                   self.now, prev_prompt_blocks=prev, warmed=ss.warmed,
                                   program_arrival=s.program_arrival, pinned=pinned))
        ss.warmed = False

    def finish_turn(self, r: Request) -> None:
        ss = self.live[r.sid]
        t = ss.replay.turns[ss.turn]
        self.rec.requests.append(self.request_record(r))
        if hasattr(self.policy, "observe_queue_delay"):
            self.policy.observe_queue_delay(r.admitted - r.arrival)
        s = ss.state
        s.state = State.PAUSED
        s.last_used = self.now
        s.context_tokens = r.prompt_tokens + r.output_tokens
        s.program = t.program
        s.pause_start = self.now
        s.true_pause_s = t.pause_s  # None on the last turn: the session never returns
        action = self.policy.on_pause(s, self.now)
        pred = s.prediction
        ss.pause_rec = {
            "sid": s.session_id, "turn": ss.turn, "start": self.now, "program": t.program,
            "true_pause_s": t.pause_s, "p10": pred.p10 if pred else None, "p50": pred.p50 if pred else None,
            "p90": pred.p90 if pred else None, "confident": bool(pred and pred.confident), "action": action.value,
        }
        self.rec.pauses.append(ss.pause_rec)
        if action is Action.OFFLOAD and self.cpu is not None:
            self.leave_gpu(ss, State.OFFLOADED)
        elif action is not Action.KEEP:
            if self.cpu is not None:
                self.cpu.drop_owner(s.session_id)
            self.leave_gpu(ss, State.DISCARDED)
        if t.pause_s is not None:
            self.push(self.now + t.pause_s, TURN, "turn", ss)

    def finish_warmup(self, r: Request) -> None:
        ss = self.live[r.sid]
        ss.warmup_rec.update(loaded=self.now, in_time=ss.state.state is State.OFFLOADED)
        if ss.state.state is State.OFFLOADED:  # the session has not returned meanwhile
            ss.state.state = State.PAUSED
            ss.warmed = True

    def leave_gpu(self, ss: SimSession, state: State) -> None:
        self.gpu.evict_owner(ss.state.session_id)
        ss.state.state = state
        if state is State.OFFLOADED and self.cfg.prefetch:
            t = self.policy.prefetch_at(ss.state)
            if t is not None and math.isfinite(t):
                ss.prefetch_token += 1
                self.push(max(t, self.now), PREFETCH, "prefetch", (ss, ss.prefetch_token))

    def make_room(self, need: int, now: float) -> int:
        """Eviction path: whole paused sessions in policy order, then the shared prefix by LRU."""
        paused = [ss.state for ss in self.live.values()
                  if ss.state.state is State.PAUSED and self.gpu.cached_of(ss.state.session_id)]
        freed = 0
        for s in self.policy.eviction_order(paused, now):
            if freed >= need:
                break
            freed += self.gpu.cached_of(s.session_id)
            self.leave_gpu(self.live[s.session_id], State.OFFLOADED if self.cpu is not None else State.DISCARDED)
        if freed < need:
            freed += self.gpu.evict_lru(need - freed, owners=[None])
        return freed

    def locate(self, ss: SimSession) -> str:
        """Where most of the returning session's previous prompt is: gpu, cpu or gone (a peek, no LRU touch)."""
        h = ss.hashes[ss.turn]
        if not h:
            return "gone"
        if self.gpu.hit(h) >= len(h) - 1:
            return "gpu"
        whole_chunks = len(h) // 16 * 16  # the CPU tier only ever holds whole 256-token chunks
        if self.cpu is not None and whole_chunks and self.cpu.hit_blocks(h, touch=False) >= whole_chunks:
            return "cpu"
        return "gone"

    # --- records -------------------------------------------------------------------------------------
    def request_record(self, r: Request) -> dict:
        reusable = min(r.prev_prompt_blocks * BLOCK, r.prompt_tokens - 1) if r.turn else 0
        re_prefilled = max(0, reusable - r.gpu_hit - r.cpu_load)
        steps = self.server.steps
        return {
            "sid": r.sid, "turn": r.turn, "arrival": r.arrival, "admitted": r.admitted,
            "first_token": r.first_token, "finish": r.finish, "ttft": r.first_token - r.arrival,
            "prompt_tokens": r.prompt_tokens, "output_tokens": r.output_tokens,
            "gpu_hit": r.gpu_hit, "cpu_load": r.cpu_load, "recomputed": r.recomputed,
            "re_prefilled": re_prefilled,
            "re_prefill_s": steps.prefill_s(r.gpu_hit + r.cpu_load, re_prefilled),
            "warmed": r.warmed,
        }
