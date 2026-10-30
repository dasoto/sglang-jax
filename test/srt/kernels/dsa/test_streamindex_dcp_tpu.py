"""StreamIndex kernels in DCP mode: per-rank local top-k merged across ranks must
select the same entries as the single-rank kernel over the whole cache.

Page j of a sequence lives on rank j % dcp at local page (g - 1) // dcp + 1, the
layout of DSASparseAttentionBackend with --dcp-size. TPU-only (same constraint
as test_streamindex_topk.py).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.streamindex_topk import (
    streamindex_page_topk,
    streamindex_topk,
)

if jax.default_backend() != "tpu":
    pytest.skip("streamindex kernels require TPU", allow_module_level=True)

PS, H, D, DCP = 128, 32, 128, 4


def _layout(rng, seq_lens, pages_per_seq, num_pages):
    """Global fixed-stride page table with strict round-robin ownership."""
    free = {
        r: [g for g in range(1, num_pages) if (g - 1) % DCP == r] for r in range(DCP)
    }
    for r in free:
        rng.shuffle(free[r])
    table = np.zeros((len(seq_lens), pages_per_seq), np.int32)
    for s, n in enumerate(seq_lens):
        for j in range(-(-n // PS)):
            table[s, j] = free[j % DCP].pop()
    return table


def _local(cache, table, rank):
    num_pages = cache.shape[0]
    local_cache = np.zeros(
        (-(-(num_pages - 1) // DCP) + 1,) + cache.shape[1:], cache.dtype
    )
    for g in range(1, num_pages):
        if (g - 1) % DCP == rank:
            local_cache[(g - 1) // DCP + 1] = cache[g]
    lpps = -(-table.shape[1] // DCP)
    j = np.arange(lpps) * DCP + rank
    g = np.where(
        j[None, :] < table.shape[1], table[:, np.minimum(j, table.shape[1] - 1)], 0
    )
    local_table = np.where(g > 0, (g - 1) // DCP + 1, 0).astype(np.int32)
    return jnp.asarray(local_cache), jnp.asarray(local_table.reshape(-1))


def _merge(per_rank, k):
    vals = jnp.concatenate([v for v, _ in per_rank], axis=1)
    idx = jnp.concatenate([i for _, i in per_rank], axis=1)
    top, pos = jax.lax.top_k(vals, k)
    return np.asarray(
        jnp.where(top > -jnp.inf, jnp.take_along_axis(idx, pos, axis=1), -1)
    )


def _same_sets(a, b):
    a = np.sort(np.asarray(a), axis=1)
    b = np.sort(np.asarray(b), axis=1)
    np.testing.assert_array_equal(a, b)


def _inputs(rng, t, num_pages):
    q = jnp.asarray(rng.standard_normal((t, H, D)), jnp.bfloat16)
    w = jnp.asarray(np.abs(rng.standard_normal((t, H))), jnp.float32)
    cache = (rng.standard_normal((num_pages, PS // 2, 2, D)) * 0.5).astype(np.float32)
    return q, w, np.asarray(jnp.asarray(cache, jnp.bfloat16))


@pytest.mark.parametrize(
    "q_lens, seq_lens", [([64, 200], [700, 1500]), ([128], [3000])]
)
def test_page_topk_dcp_merge(q_lens, seq_lens):
    rng = np.random.default_rng(1)
    num_pages, pps, k_pages = 80, 32, 6
    q, w, cache = _inputs(rng, sum(q_lens), num_pages)
    table = _layout(rng, seq_lens, pps, num_pages)
    sl = jnp.asarray(seq_lens, jnp.int32)
    cu_q = jnp.asarray(np.cumsum([0] + q_lens), jnp.int32)
    n = jnp.int32(len(seq_lens))
    ref = streamindex_page_topk(
        q,
        w,
        jnp.asarray(cache),
        sl,
        jnp.asarray(table.reshape(-1)),
        cu_q,
        n,
        k_pages=k_pages,
        num_queries_per_block=64,
    )
    per_rank = []
    for r in range(DCP):
        lc, lt = _local(cache, table, r)
        per_rank.append(
            streamindex_page_topk(
                q,
                w,
                lc,
                sl,
                lt,
                cu_q,
                n,
                k_pages=k_pages,
                num_queries_per_block=64,
                dcp_size=DCP,
                dcp_rank=jnp.int32(r),
                return_values=True,
            )
        )
    _same_sets(_merge(per_rank, k_pages), ref)


def test_token_topk_dcp_merge():
    rng = np.random.default_rng(2)
    num_pages, pps, k = 80, 32, 256
    seq_lens = [700, 129, 2500, 3900]
    b = len(seq_lens)
    q, w, cache = _inputs(rng, b, num_pages)
    table = _layout(rng, seq_lens, pps, num_pages)
    sl = jnp.asarray(seq_lens, jnp.int32)
    cu_q = jnp.arange(b + 1, dtype=jnp.int32)
    dist = jnp.asarray([b, b, b], jnp.int32)
    kw = dict(
        k=k, compression_ratio=1, num_kv_pages_per_block=64, num_queries_per_block=1
    )
    ref = streamindex_topk(
        q, w, jnp.asarray(cache), sl, jnp.asarray(table.reshape(-1)), cu_q, dist, **kw
    )
    per_rank = []
    for r in range(DCP):
        lc, lt = _local(cache, table, r)
        per_rank.append(
            streamindex_topk(
                q,
                w,
                lc,
                sl,
                lt,
                cu_q,
                dist,
                dcp_size=DCP,
                dcp_rank=jnp.int32(r),
                return_values=True,
                **kw,
            )
        )
    _same_sets(_merge(per_rank, k), ref)
