"""kvtier's split attention for mixed steps (vllm fork, triton_attn.py) matches vLLM's unified Triton kernel."""

import pytest

torch = pytest.importorskip("torch")

BS = 16  # vLLM block size


def make_batch(q_lens, k_lens, hq, hkv, hs, seed=0):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention  # noqa: F401

    g = torch.Generator(device="cuda").manual_seed(seed)
    blocks_per = [-(-k // BS) for k in k_lens]
    nb = sum(blocks_per) + 3
    # cache in vLLM's layout: logical (B, H, N, 2*hs), seen by the kernel as (B, N, H, hs) views
    kv = torch.randn(nb, hkv, BS, 2 * hs, generator=g, device="cuda", dtype=torch.float16).transpose(1, 2)
    k_cache, v_cache = kv.split(hs, dim=-1)
    perm = torch.randperm(nb, generator=g, device="cuda").to(torch.int32)  # scattered blocks
    bt = torch.zeros(len(q_lens), max(blocks_per), dtype=torch.int32, device="cuda")
    start = 0
    for i, b in enumerate(blocks_per):
        bt[i, :b] = perm[start : start + b]
        start += b
    q = torch.randn(sum(q_lens), hq, hs, generator=g, device="cuda", dtype=torch.float16)
    starts = [0]
    for n in q_lens:
        starts.append(starts[-1] + n)
    return q, k_cache, v_cache, bt, starts


def reference(q, k_cache, v_cache, bt, starts, k_lens, scale):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention

    out = torch.empty_like(q)
    n = len(k_lens)
    ones = torch.ones(n, k_cache.shape[2], device="cuda")
    unified_attention(q=q, k=k_cache, v=v_cache, out=out,
                      cu_seqlens_q=torch.tensor(starts, dtype=torch.int32, device="cuda"),
                      max_seqlen_q=max(b - a for a, b in zip(starts, starts[1:])),
                      seqused_k=torch.tensor(k_lens, dtype=torch.int32, device="cuda"), max_seqlen_k=max(k_lens),
                      softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=bt, softcap=0,
                      q_descale=None, k_descale=ones, v_descale=ones)
    return out


@pytest.mark.gpu
@pytest.mark.parametrize("hs", [64, 128])
def test_split_matches_unified_kernel(hs):
    from vllm.v1.attention.backends.triton_attn import build_sdpa_split, sdpa_split_attention

    hq, hkv, segs, thr = 14, 2, 16, 64
    # decodes with assorted contexts, a prefill chunk on cached context, a fresh prompt, a 1-block prompt
    q_lens = [1, 37, 1, 1, 300, 1, 16]
    k_lens = [700, 37 + 2000, 5, 4096, 300, 1234, 16]
    q, kc, vc, bt, starts = make_batch(q_lens, k_lens, hq, hkv, hs)
    scale = hs ** -0.5
    want = reference(q, kc, vc, bt, starts, k_lens, scale)

    split = build_sdpa_split(starts, k_lens, torch.tensor(k_lens, dtype=torch.int32, device="cuda"), bt, BS,
                             torch.float16)
    assert [p.q_end - p.q_start for p in split.prefills] == [37, 300, 16]
    assert split.dec_tokens.tolist() == [0, 38, 39, 340]
    got = torch.full_like(q, float("nan"))
    hp = 1 << (hs - 1).bit_length()
    ones = torch.ones(len(k_lens), hkv, device="cuda")
    sdpa_split_attention(q, kc, vc, got, split, scale, ones, ones, thr, segs,
                         torch.empty(thr, hq, segs, hp, device="cuda"), torch.empty(thr, hq, segs, device="cuda"),
                         torch.empty(thr, hq, segs, device="cuda"))
    assert not got.isnan().any()  # every token row written
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-2)


@pytest.mark.gpu
def test_split_with_only_prefills():
    from vllm.v1.attention.backends.triton_attn import build_sdpa_split, sdpa_split_attention

    q_lens, k_lens = [64, 512], [64 + 1000, 512]
    q, kc, vc, bt, starts = make_batch(q_lens, k_lens, 14, 2, 64, seed=1)
    want = reference(q, kc, vc, bt, starts, k_lens, 0.125)
    split = build_sdpa_split(starts, k_lens, torch.tensor(k_lens, dtype=torch.int32, device="cuda"), bt, BS,
                             torch.float16)
    assert split.dec_tokens.numel() == 0
    got = torch.full_like(q, float("nan"))
    ones = torch.ones(2, 2, device="cuda")
    sdpa_split_attention(q, kc, vc, got, split, 0.125, ones, ones, 64, 16, None, None, None)
    torch.testing.assert_close(got, want, atol=2e-3, rtol=2e-2)
