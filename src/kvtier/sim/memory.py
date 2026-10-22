"""KV memory: GPU block pool with prefix caching, and the LMCache-style CPU tier.

GPU blocks are keyed by their hash. A block is in use while referenced by a running request,
cached when unreferenced (reusable and evictable), and gone once evicted. Partial blocks (the
last, unfilled block of a request) are anonymous: counted while in use, freed at finish.
Each hashed block has an owner session, except the cross-session shared prefix (owner None).
"""

from collections import OrderedDict
from dataclasses import dataclass

from kvtier.sessions import BLOCK

CHUNK_BLOCKS = 256 // BLOCK  # LMCache stores 256-token chunks


@dataclass
class Block:
    owner: str | None
    refs: int = 0
    freed_at: float = 0.0


class GpuPool:
    def __init__(self, capacity_blocks: int):
        self.capacity = capacity_blocks
        self.blocks: dict = {}  # hash -> Block
        self.by_owner: dict[str | None, set] = {}
        self.anon = 0

    @property
    def used(self) -> int:
        return len(self.blocks) + self.anon

    @property
    def free(self) -> int:
        return self.capacity - self.used

    def in_use(self) -> int:
        return self.anon + sum(1 for b in self.blocks.values() if b.refs > 0)

    def hit(self, hashes: list) -> int:
        """Number of leading blocks of `hashes` resident on GPU."""
        n = 0
        while n < len(hashes) and hashes[n] in self.blocks:
            n += 1
        return n

    def ref(self, hashes: list) -> None:
        """Reference resident blocks (cache hits)."""
        for h in hashes:
            self.blocks[h].refs += 1

    def add(self, hashes: list, owners: list) -> None:
        """Allocate hashed blocks, referenced once (an already resident block is just referenced).
        Caller checks free space."""
        for h, owner in zip(hashes, owners):
            if h in self.blocks:
                self.blocks[h].refs += 1
                continue
            self.blocks[h] = Block(owner, refs=1)
            self.by_owner.setdefault(owner, set()).add(h)

    def release(self, hashes: list, anon: int, now: float) -> None:
        for h in hashes:
            b = self.blocks[h]
            b.refs -= 1
            if b.refs == 0:
                b.freed_at = now
        self.anon -= anon

    def cached_of(self, owner: str) -> int:
        return sum(1 for h in self.by_owner.get(owner, ()) if self.blocks[h].refs == 0)

    def _drop(self, h) -> None:
        b = self.blocks.pop(h)
        owned = self.by_owner[b.owner]
        owned.discard(h)
        if not owned:
            del self.by_owner[b.owner]

    def evict_owner(self, owner: str) -> int:
        """Evict all of an owner's unreferenced blocks; returns blocks freed."""
        victims = [h for h in self.by_owner.get(owner, ()) if self.blocks[h].refs == 0]
        for h in victims:
            self._drop(h)
        return len(victims)

    def evict_lru(self, n: int, owners=None) -> int:
        """Evict up to n unreferenced blocks, oldest freed first; `owners` limits whose (None = any)."""
        pool = self.blocks.items() if owners is None else (
            (h, self.blocks[h]) for o in owners for h in self.by_owner.get(o, ()))
        victims = sorted(((b.freed_at, h) for h, b in pool if b.refs == 0), key=lambda x: x[0])[:n]
        for _, h in victims:
            self._drop(h)
        return len(victims)


class CpuTier:
    """LMCache CPU backend: full 256-token prompt chunks, LRU at capacity, keyed by the chunk's last block hash."""

    def __init__(self, capacity_tokens: int):
        self.capacity = capacity_tokens // (CHUNK_BLOCKS * BLOCK)
        self.chunks: OrderedDict = OrderedDict()  # key -> owner

    @staticmethod
    def keys(hashes: list) -> list:
        return [hashes[(i + 1) * CHUNK_BLOCKS - 1] for i in range(len(hashes) // CHUNK_BLOCKS)]

    def store(self, hashes: list, owner: str, shared_blocks: int) -> None:
        for i, k in enumerate(self.keys(hashes)):
            if k in self.chunks:
                self.chunks.move_to_end(k)
            else:
                self.chunks[k] = None if (i + 1) * CHUNK_BLOCKS <= shared_blocks else owner
        while len(self.chunks) > self.capacity:
            self.chunks.popitem(last=False)

    def hit_blocks(self, hashes: list, touch: bool = True) -> int:
        """Leading blocks of `hashes` available on CPU (whole chunks only); a lookup touches them for LRU."""
        n = 0
        for k in self.keys(hashes):
            if k not in self.chunks:
                break
            if touch:
                self.chunks.move_to_end(k)
            n += CHUNK_BLOCKS
        return n

    def drop_owner(self, owner: str) -> None:
        for k in [k for k, o in self.chunks.items() if o == owner]:
            del self.chunks[k]
