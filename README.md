# agent-kv-cache

## Smarter GPU memory for AI coding agents.

A coding agent talks to an LLM in turns. The model replies, then the agent runs a tool and then it comes back for the next turn. While the tool runs, the agent's KV cache sits on the GPU doing nothing. When GPU memory fills up, the server has to throw some of that memory out. Normally it picks the session that's been idle longest. It has no idea which agent is about to come back.

## What this project does

It makes that choice tool-aware. It learns how long each kind of tool call usually takes. When memory runs out, it evicts the agent that's likely to stay away the longest. Agents that are about to return keep their cache, so their next turn starts fast.

The guess updates as the pause goes on. An agent that's been gone longer than usual moves to the front of the line.

## Results so far

All numbers below come from a simulator, not real hardware yet.

- When GPU memory is the bottleneck, the policy serves up to 11% more requests on time than vLLM
  with LMCache. It never did worse in any test.
- A perfect guesser that knows the future gets up to 34%. We capture a third to half of that.
- We rebuilt [Continuum](https://arxiv.org/abs/2511.02230), a 2025 research system for the same
  problem. Its gains come from how it orders requests, not from how it handles memory. Our policy
  plus its ordering did best: up to 70% more requests on time.
- Moving memory around early, by offloading or prefetching, didn't help. It lost even with
  perfect knowledge of pause lengths.

## How it works

1. **Record.** Claude drives [mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) on
   SWE-bench tasks. We log every prompt, reply and tool pause. We have 70 traces so far.
2. **Predict.** A small model estimates how much longer each tool call will take, from past
   calls to the same tool.
3. **Simulate.** A fast simulator of a vLLM-style server replays the traces under different
   policies. One run takes about 2 seconds.
4. **Serve.** The policy plugs into vLLM's scheduler. This part is in progress.


## Running it

You need Python 3.11, [uv](https://docs.astral.sh/uv/) and an NVIDIA GPU for serving.

```bash
git clone --recurse-submodules https://github.com/mihirrd/agent-kv-cache.git
cd agent-kv-cache
uv sync
uv run pytest -m "not gpu"
uv run python -m kvtier.sim run --policy remaining-evict always-offload --load 22 --speedup 4
```
