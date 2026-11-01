"""C1: vLLM's block pool evicts paused sessions' cached blocks in the policy's order (fork module
vllm/v1/core/kvtier_sessions.py). Runs vLLM's real BlockPool on CPU; no GPU or server needed."""

import time

import pytest

pytest.importorskip("vllm")

from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id  # noqa: E402
from vllm.v1.core.kvtier_sessions import SessionEviction  # noqa: E402


def make_pool(n_blocks: int, oracle: bool = True) -> BlockPool:
    pool = BlockPool(num_gpu_blocks=n_blocks + 1, enable_caching=True, hash_block_size=16)  # +1: null block
    pool.kvtier = SessionEviction(oracle=oracle)
    return pool


def finish(pool: BlockPool, sid: str, n: int, pause_s: float | None, program: str = "pytest"):
    """One request of session `sid`: arrives, fills n cached blocks, finishes, frees them like vLLM."""
    pool.kvtier.on_arrival(sid, time.monotonic())
    blocks = pool.get_new_blocks(n)
    for i, b in enumerate(blocks):
        pool._insert_block_hash(make_block_hash_with_group_id(BlockHash(f"{sid}/{i}".encode()), 0), b, 16)
    pool.kvtier.on_finish(sid, program, pause_s, blocks, time.monotonic())
    pool.free_blocks(reversed(blocks))  # vLLM frees tail first
    return blocks


def cached(blocks) -> list[bool]:
    return [b.block_hash is not None for b in blocks]


def test_free_uncached_blocks_are_used_before_any_eviction():
    pool = make_pool(8)
    a = finish(pool, "a", 3, pause_s=5.0)
    pool.get_new_blocks(5)
    assert cached(a) == [True, True, True]
    assert pool.kvtier.evicted_by_policy == pool.kvtier.evicted_by_lru == 0


def test_oracle_evicts_longest_pause_first_tail_first():
    pool = make_pool(12)
    a = finish(pool, "a", 3, pause_s=10.0)
    b = finish(pool, "b", 3, pause_s=100.0)
    c = finish(pool, "c", 3, pause_s=50.0)
    # LRU would evict a first; the oracle takes b (back last), tail blocks first
    pool.get_new_blocks(3 + 2)
    assert cached(b) == [True, False, False]
    assert cached(a) == cached(c) == [True, True, True]
    pool.get_new_blocks(2)
    assert cached(b) == [False] * 3 and cached(c) == [True, True, False] and cached(a) == [True] * 3
    assert pool.kvtier.evicted_by_policy == 4 and pool.kvtier.evicted_by_lru == 0


def test_ended_session_goes_first():
    pool = make_pool(6)
    a = finish(pool, "a", 3, pause_s=1000.0)
    b = finish(pool, "b", 3, pause_s=None)  # last turn: the session never returns
    pool.get_new_blocks(3)
    assert cached(b) == [False] * 3 and cached(a) == [True] * 3


def test_returned_sessions_are_not_victims_and_lru_is_the_fallback():
    pool = make_pool(6)
    a = finish(pool, "a", 3, pause_s=10.0)
    b = finish(pool, "b", 3, pause_s=100.0)
    pool.kvtier.on_arrival("b", time.monotonic())  # b is back (waiting): its blocks are about to be reused
    pool.get_new_blocks(4)
    # a goes by policy; the last block comes from the LRU end of the queue (b's freed tail block)
    assert cached(a) == [False] * 3 and cached(b) == [True, True, False]
    assert pool.kvtier.evicted_by_policy == 3 and pool.kvtier.evicted_by_lru == 1


def test_blocks_the_next_turn_dropped_go_first():
    pool = make_pool(7)
    a1 = finish(pool, "a", 3, pause_s=1.0)  # turn 1: prompt p0 p1, reply block r
    finish(pool, "b", 3, pause_s=100.0)
    # turn 2 reuses p0 p1 but not r (the next prompt holds a different reply), adds one block
    pool.kvtier.on_arrival("a", time.monotonic())
    pool.touch(a1[:2])
    new = pool.get_new_blocks(1)
    pool.kvtier.on_finish("a", "grep", 1.0, a1[:2] + new, time.monotonic())
    pool.free_blocks(reversed(a1[:2] + new))
    assert pool.kvtier.orphans == {a1[2].block_id: a1[2]}
    # after the uncached new block, r goes first, before b, whose pause is far longer
    pool.get_new_blocks(2)
    assert cached(a1) == [True, True, False]
    assert pool.kvtier.evicted_orphans == 1 and pool.kvtier.evicted_by_policy == 0


def test_blocks_in_use_are_skipped():
    pool = make_pool(6)
    a = finish(pool, "a", 3, pause_s=100.0)
    finish(pool, "b", 3, pause_s=10.0)
    pool.touch([a[0]])  # a's prefix block is shared with a running request
    pool.get_new_blocks(2)
    assert cached(a) == [True, False, False] and a[0].ref_cnt == 1
    assert pool.kvtier.evicted_by_policy == 2


def test_reused_block_is_not_evicted_twice():
    pool = make_pool(6)
    a = finish(pool, "a", 3, pause_s=100.0)
    finish(pool, "b", 3, pause_s=10.0)
    taken = pool.get_new_blocks(1)  # a's tail block, now owned by a new request
    assert taken == [a[2]]
    pool.free_blocks(taken)  # uncached again: freed to the front
    pool.get_new_blocks(3)
    assert cached(a) == [False, False, False]


def test_predicted_time_left_orders_eviction():
    pool = make_pool(6, oracle=False)
    for _ in range(5):  # history: pytest pauses last ~60 s, grep pauses ~0.1 s
        pool.kvtier.policy.predictor.observe("other", "pytest", 60.0)
        pool.kvtier.policy.predictor.observe("other", "grep", 0.1)
    test = finish(pool, "t", 3, pause_s=None, program="pytest")
    grep = finish(pool, "g", 3, pause_s=None, program="grep")
    pool.get_new_blocks(3)
    assert cached(test) == [False] * 3 and cached(grep) == [True] * 3
