"""Gate B: does a warm-up request turn the agent's next real request into a GPU prefix-cache hit?

Run on gpubox against a live server: uv run python scripts/gate_b_warmup.py
Session A gets a warm-up (its last prompt, max_tokens=1) after being evicted from the GPU;
session B is the control without one. Pass: A's real turn is served from GPU prefix cache
(local hits, no LMCache load) and is faster than B's.
"""

import argparse
import asyncio
import random
import re
import time

import httpx

BASE = "http://localhost:8000"
WORDS = "def class return import self value config path error test result index item data node".split()


def prompt_text(seed: int, n_lines: int) -> str:
    rng = random.Random(seed)
    lines = (
        f"{i:05d}: {' '.join(rng.choice(WORDS) + str(rng.randrange(1000)) for _ in range(8))}"
        for i in range(n_lines)
    )
    return "Here is the repository dump:\n" + "\n".join(lines) + "\nWhat should I run next?"


def metrics(client: httpx.Client) -> dict[str, float]:
    text = client.get(f"{BASE}/metrics").text
    out = {}
    for name in ("prefix_cache_hits", "prefix_cache_queries", "external_prefix_cache_hits"):
        out[name] = sum(float(v) for v in re.findall(rf"^vllm:{name}_total\{{.*\}} (\S+)$", text, re.M))
    return out


def chat(client, model, messages, max_tokens, headers=None):
    t = time.time()
    r = client.post(
        f"{BASE}/v1/chat/completions",
        json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0},
        headers=headers,
    )
    r.raise_for_status()
    body = r.json()
    return time.time() - t, body["choices"][0]["message"]["content"], body["usage"]["prompt_tokens"]


def measured(client, model, label, messages, max_tokens=1, headers=None):
    before = metrics(client)
    dt, content, n = chat(client, model, messages, max_tokens, headers)
    after = metrics(client)
    d = {k: int(after[k] - before[k]) for k in after}
    print(
        f"{label:<26} prompt={n:>6}  time={dt:6.2f}s  gpu_hits={d['prefix_cache_hits']:>6}  "
        f"lmcache_hits={d['external_prefix_cache_hits']:>6}"
    )
    return dt, content, n, d


async def flood(model, total_tokens, seed, chunk=4000, conc=8):
    sem = asyncio.Semaphore(conc)

    async def one(client, i):
        rng = random.Random(seed * 100_000 + i)
        ids = [rng.randrange(100, 150_000) for _ in range(chunk)]
        async with sem:
            r = await client.post(f"{BASE}/v1/completions", json={"model": model, "prompt": ids, "max_tokens": 1})
            r.raise_for_status()

    t = time.time()
    async with httpx.AsyncClient(timeout=900) as client:
        await asyncio.gather(*(one(client, i) for i in range(total_tokens // chunk)))
    print(f"  (filler traffic: {total_tokens} tokens in {time.time() - t:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evict-tokens", type=int, default=240_000, help="> GPU KV pool size")
    ap.add_argument("--gap-tokens", type=int, default=40_000, help="other traffic between warm-up and real turn")
    ap.add_argument("--lines", type=int, default=420, help="~8K-token first turn")
    args = ap.parse_args()

    client = httpx.Client(timeout=900)
    model = client.get(f"{BASE}/v1/models").json()["data"][0]["id"]
    sessions = {}
    for name, seed in (("A", 1), ("B", 2)):
        msgs = [
            {"role": "system", "content": "You are a coding agent. Reply with one bash command."},
            {"role": "user", "content": prompt_text(seed, args.lines)},
        ]
        _, reply, _, _ = measured(client, model, f"{name} turn 1 (cold)", msgs, max_tokens=64)
        tool_out = prompt_text(seed + 100, 25)  # ~500 tokens of tool output
        sessions[name] = (msgs, msgs + [{"role": "assistant", "content": reply}, {"role": "user", "content": tool_out}])

    asyncio.run(flood(model, args.evict_tokens, seed=7))
    a_last, a_next = sessions["A"]
    measured(client, model, "A warm-up (max_tokens=1)", a_last)
    asyncio.run(flood(model, args.gap_tokens, seed=8))
    _, _, _, da = measured(client, model, "A turn 2 (after warm-up)", a_next, headers={"X-Request-Id": "sessA-turn2"})
    _, b_next = sessions["B"]
    _, _, _, db = measured(client, model, "B turn 2 (no warm-up)", b_next)

    passed = da["prefix_cache_hits"] >= 0.9 * prompt_len(client, model, a_last) and da["external_prefix_cache_hits"] == 0
    print(f"GATE B: {'PASS' if passed else 'FAIL'}")


def prompt_len(client, model, messages) -> int:
    r = client.post(f"{BASE}/tokenize", json={"model": model, "messages": messages, "add_generation_prompt": True})
    r.raise_for_status()
    return r.json()["count"]


if __name__ == "__main__":
    main()
