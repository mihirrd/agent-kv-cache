"""vLLM V1-style server: FCFS queue, continuous batching with chunked prefill, one step at a time.

Simplification (C7 spec): a request's blocks for prompt + output are allocated when it is
admitted, so running requests never run out of blocks and there is no preemption; a request that
cannot get blocks waits at the head of the queue.
"""

import itertools
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

from kvtier.sessions import BLOCK
from kvtier.sim.memory import CpuTier, GpuPool
from kvtier.sim.timing import StepModel

_out_ids = itertools.count()


@dataclass(eq=False)
class Request:
    sid: str
    turn: int
    hashes: list
    prompt_tokens: int
    output_tokens: int
    arrival: float
    warmup: bool = False
    prev_prompt_blocks: int = 0  # blocks of the session's previous prompt (reusable prefix)
    warmed: bool = False  # a prefetch warm-up completed during the pause before this turn
    program_arrival: float = 0.0  # arrival of the session's first request (program-level FCFS)
    pinned: bool = False  # the session's KV was still TTL-pinned when this request arrived
    gpu_hit: int = 0  # tokens served from GPU prefix cache
    cpu_load: int = 0  # tokens reloaded from the CPU tier
    prefilled: int = 0  # prompt tokens done, cached ones included
    decoded: int = 0
    admitted: float | None = None
    first_token: float | None = None
    finish: float | None = None
    out_keys: list = field(default_factory=list)  # hashed blocks of generated tokens (never reused)
    anon: int = 0

    @property
    def recomputed(self) -> int:
        return self.prompt_tokens - self.gpu_hit - self.cpu_load


@dataclass
class StepPlan:
    decodes: list
    chunks: list  # (request, n tokens)
    admitted: list
    duration: float


class Server:
    def __init__(self, gpu: GpuPool, cpu: CpuTier | None, steps: StepModel, shared_blocks: int,
                 make_room: Callable[[int, float], int], queue_key: Callable[[Request], tuple] | None = None):
        self.gpu, self.cpu, self.steps = gpu, cpu, steps
        self.shared_blocks = shared_blocks
        self.make_room = make_room  # (blocks needed, now) -> blocks freed; the policy's eviction path
        self.queue_key = queue_key  # waiting-queue order; None = FCFS by request arrival (vLLM)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def submit(self, r: Request) -> None:
        self.waiting.append(r)

    def admit(self, r: Request, now: float) -> bool:
        hit_blocks = self.gpu.hit(r.hashes)
        hit_tokens = min(hit_blocks * BLOCK, r.prompt_tokens - 1)  # vLLM always computes the last token
        hit_blocks = hit_tokens // BLOCK
        cpu_tokens = 0
        if self.cpu is not None:
            cpu_tokens = min(self.cpu.hit_blocks(r.hashes) * BLOCK, r.prompt_tokens - 1)
        load = max(0, cpu_tokens - hit_tokens) // BLOCK * BLOCK

        total = r.prompt_tokens + r.output_tokens
        new_prompt = r.hashes[hit_blocks:]
        n_out = total // BLOCK - len(r.hashes)  # full blocks of generated tokens
        anon = -(-total // BLOCK) - total // BLOCK  # the partial last block, if any
        need = len(new_prompt) + n_out + anon
        if need > self.gpu.capacity:
            raise RuntimeError(f"request needs {need} blocks, pool has {self.gpu.capacity}")

        self.gpu.ref(r.hashes[:hit_blocks])  # pin hits before evicting anything
        if need > self.gpu.free:
            self.make_room(need - self.gpu.free, now)
        if need > self.gpu.free:
            self.gpu.evict_lru(need - self.gpu.free)  # last resort: any unreferenced block
        if need > self.gpu.free:
            self.gpu.release(r.hashes[:hit_blocks], 0, now)
            return False

        owners = [None if hit_blocks + i < self.shared_blocks else r.sid for i in range(len(new_prompt))]
        self.gpu.add(new_prompt, owners)
        r.out_keys = [("out", next(_out_ids)) for _ in range(n_out)]
        self.gpu.add(r.out_keys, [r.sid] * n_out)
        self.gpu.anon += anon
        r.anon = anon
        r.gpu_hit, r.cpu_load = hit_tokens, load
        r.prefilled = hit_tokens + load
        r.admitted = now
        return True

    def plan(self, now: float) -> StepPlan | None:
        budget = self.steps.cal.prefill_budget_tokens
        if self.queue_key is not None and len(self.waiting) > 1:
            self.waiting = deque(sorted(self.waiting, key=self.queue_key))
        decodes = [r for r in self.running if r.prefilled >= r.prompt_tokens]
        chunks, admitted = [], []
        for r in self.running:
            if budget and r.prefilled < r.prompt_tokens:
                k = min(budget, r.prompt_tokens - r.prefilled)
                chunks.append((r, k))
                budget -= k
        while budget and self.waiting:
            r = self.waiting[0]
            if not self.admit(r, now):
                break
            self.waiting.popleft()
            self.running.append(r)
            admitted.append(r)
            k = min(budget, r.prompt_tokens - r.prefilled)
            chunks.append((r, k))
            budget -= k
        if not decodes and not chunks:
            return None
        duration = (
            self.steps.decode_s(len(decodes))
            + sum(self.steps.prefill_s(r.prefilled, k) for r, k in chunks)
            + sum(self.steps.reload_s(r.cpu_load) for r in admitted)
        )
        return StepPlan(decodes, chunks, admitted, duration)

    def apply(self, plan: StepPlan, end: float) -> list[Request]:
        """Advance the requests in the step; returns those that finished."""
        done = []
        for r in plan.decodes:
            r.decoded += 1
            if r.decoded >= r.output_tokens:
                done.append(r)
        for r, k in plan.chunks:
            r.prefilled += k
            if r.prefilled >= r.prompt_tokens:
                r.first_token = end  # the step that completes prefill emits the first token
                r.decoded = 1
                if r.decoded >= r.output_tokens:
                    done.append(r)
        for r in done:
            r.finish = end
            self.running.remove(r)
            self.gpu.release(r.hashes + r.out_keys, r.anon, end)
            if self.cpu is not None:
                self.cpu.store(r.hashes, r.sid, self.shared_blocks)
        return done
