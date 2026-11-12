"""CPU check of the DSA backend's rows-first mixed path under DCP.

SGLANG_JAX_MLA_DECODE_ROWS_FIRST puts each DP rank's one-token sequences first
(distribution (d, d, n)); the DSA backend then permutes the tokens into that order,
scores the one-token rows with the batched decode indexer kernel (max-pooled to
pages) and the prefill chunks with the page kernel from sequence d on, and restores
the caller's token order. The page selection, and so the attention output, must equal
the all-mixed layout's, for one-token rows before, between and after prefill chunks.

The Pallas TPU indexer kernels are replaced by DCP-aware jnp references with the same
contracts (seq ranges, local page tables, returned ids); the sparse-prefill attention
kernel runs in interpret mode. Contexts span more pages than the page budget, so a
different selection changes the output.
"""

import functools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention import dsa_sparse_backend as dsa_mod
from sgl_jax.srt.layers.attention.dsa_sparse_backend import DSASparseAttentionBackend
from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

ROWS_FIRST_ENV = "SGLANG_JAX_MLA_DECODE_ROWS_FIRST"


def _token_scores(q, w, cache, sl, local_pt, cuq, s0, s1, dcp_size, dcp_rank):
    """[T, Lp * ps] indexer scores of tokens of sequences [s0, s1) over this rank's pages
    (column c: local page c // ps, global position (c // ps * dcp + rank) * ps + c % ps);
    -inf for masked keys and rows outside the range."""
    T = q.shape[0]
    S = sl.shape[0]
    Lp = local_pt.shape[0] // S
    ps = cache.shape[1] * cache.shape[2]
    keys = cache.reshape(cache.shape[0], ps, cache.shape[-1]).astype(jnp.float32)
    t = jnp.arange(T)
    seq = jnp.clip(jnp.searchsorted(cuq[1:], t, side="right"), 0, S - 1)
    in_rng = (seq >= s0) & (seq < s1) & (t < cuq[S])
    q_len = cuq[seq + 1] - cuq[seq]
    q_pos = sl[seq] - q_len + (t - cuq[seq])
    pt = local_pt.reshape(S, Lp)[seq]  # [T, Lp]
    kk = keys[pt]  # [T, Lp, ps, D]
    qd = q.shape[-1]
    s = jnp.einsum("thd,tlpd->thlp", q.astype(jnp.float32), kk[..., :qd])
    s = (jnp.maximum(s, 0.0) * w.astype(jnp.float32)[:, :, None, None]).sum(axis=1)
    gpos = (jnp.arange(Lp)[:, None] * dcp_size + dcp_rank) * ps + jnp.arange(ps)[None, :]
    valid = (pt[:, :, None] > 0) & (gpos[None] < sl[seq][:, None, None])
    valid = valid & (gpos[None] <= q_pos[:, None, None]) & in_rng[:, None, None]
    return jnp.where(valid, s, -jnp.inf).reshape(T, Lp * ps)


def fake_page_topk(q, w, cache, sl, local_pt, cuq, num_seqs, *, k_pages, dcp_size, dcp_rank,
                   return_values, start_seq=0, **_):
    ps = cache.shape[1] * cache.shape[2]
    scores = _token_scores(q, w, cache, sl, local_pt, cuq, start_seq, num_seqs, dcp_size, dcp_rank)
    pages = scores.reshape(q.shape[0], -1, ps).max(axis=-1)
    if pages.shape[1] < k_pages:
        pages = jnp.pad(pages, ((0, 0), (0, k_pages - pages.shape[1])), constant_values=-jnp.inf)
    vals, lp = jax.lax.top_k(pages, k_pages)
    ids = jnp.where(vals > -jnp.inf, lp * dcp_size + dcp_rank, -1).astype(jnp.int32)
    assert return_values
    return vals, ids


def fake_token_topk(q, w, cache, sl, local_pt, cuq, distribution, *, dcp_size, dcp_rank,
                    return_scores=False, **_):
    assert return_scores, "only the rows-first path calls the token kernel in EXTEND mode"
    return _token_scores(q, w, cache, sl, local_pt, cuq, 0, distribution[0], dcp_size, dcp_rank)


class TestDSARowsFirstCPU(unittest.TestCase):
    page_size = 16  # the sparse-prefill qblock kernel needs page_size % 16 == 0
    num_heads = 4
    kv_lora_rank = 128
    rope = 64
    idx_heads = 4
    idx_dim = 128
    index_topk = 32  # 2 pages of 16 tokens

    @classmethod
    def setUpClass(cls):
        devices = jax.devices("cpu")
        assert len(devices) >= 2
        cls.mesh = create_device_mesh(
            ici_parallelism=[1, 2], dcn_parallelism=[1, 1], devices=devices[:2]
        )

    def _make_setup(self, total_tokens=2048):
        pool = MLATokenToKVPool(
            size=total_tokens,
            page_size=self.page_size,
            dtype=jnp.float32,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.rope,
            layer_num=1,
            mesh=self.mesh,
            dp_size=1,
            dcp_size=2,
            indexer_key_dim=self.idx_dim,
            num_indexer_layers=1,
        )
        alloc = PagedTokenToKVPoolAllocator(
            size=total_tokens, page_size=self.page_size, kvcache=pool, dp_size=1
        )
        backend = DSASparseAttentionBackend(
            index_topk=self.index_topk,
            index_head_dim=self.idx_dim,
            index_n_heads=self.idx_heads,
            skip_offset=0,
            full_slot={0: 0},
            num_attn_heads=self.num_heads,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=64,
            qk_rope_head_dim=self.rope,
            v_head_dim=64,
            page_size=self.page_size,
            mesh=self.mesh,
            dcp_size=2,
            num_kv_pages_per_block=(2, 1, 2),
            num_queries_per_block=(1, 1, 8),
            decode_batch_size=1,
        )
        layer = SimpleNamespace(layer_id=0, scaling=1.0 / np.sqrt(128.0), sliding_window_size=None,
                                logit_cap=None)
        return pool, alloc, backend, layer

    def _run(self, pool, backend, layer, seq_lens, ext_lens, cache_loc, out_loc, positions, x,
             rows_first):
        worker_batch = SimpleNamespace(
            dp_size=1,
            per_dp_bs_size=len(seq_lens),
            seq_lens=np.array(seq_lens, np.int32),
            extend_seq_lens=np.array(ext_lens, np.int32),
            cache_loc=np.array(cache_loc, np.int32),
            out_cache_loc=np.array(out_loc, np.int32),
            forward_mode=ForwardMode.EXTEND,
        )
        with mock.patch.dict(os.environ, {ROWS_FIRST_ENV: "1" if rows_first else "0"}):
            backend.forward_metadata = backend.get_forward_metadata(worker_batch)
        if rows_first:
            n_one = sum(1 for e in ext_lens if e == 1)
            dist = np.asarray(jax.device_get(backend.forward_metadata.distribution))
            self.assertEqual(list(dist), [n_one, n_one, len(seq_lens)])
        tok = NamedSharding(self.mesh, P("data"))
        put = lambda a, spec: jax.device_put(a, NamedSharding(self.mesh, spec))
        fb = SimpleNamespace(
            out_cache_loc=jax.device_put(np.array(out_loc, np.int32), tok),
            positions=jax.device_put(np.array(positions, np.int32), tok),
            forward_mode=ForwardMode.EXTEND,
        )
        with jax.set_mesh(self.mesh):
            o, fused = backend(
                put(x["q"], P("data", "tensor", None)),
                put(x["k"], P("data", None)),
                put(x["k"], P("data", None)),
                layer,
                fb,
                pool,
                q_rope=put(x["q_pe"], P("data", "tensor", None)),
                k_rope=put(x["k_pe"], P("data", None)),
                indexer_type="full",
                q_idx=put(x["q_idx"], P("data", None, None)),
                k_idx=put(x["k_idx"], P("data", None)),
                idx_weights=put(x["w"], P("data", None)),
            )
        pool.replace_buffer(([fused.kv], [fused.idx]))
        return np.asarray(jax.device_get(o))

    def test_rows_first_matches_mixed_order(self):
        rng = np.random.default_rng(20261112)
        ps = self.page_size
        attn = dsa_mod.prefill_write_and_attend_ragged_qblock
        patches = [
            mock.patch.object(dsa_mod, "streamindex_page_topk", fake_page_topk),
            mock.patch.object(dsa_mod, "streamindex_topk", fake_token_topk),
            mock.patch.object(dsa_mod, "prefill_write_and_attend_ragged_qblock",
                              functools.partial(attn, interpret=True)),
            mock.patch.object(dsa_mod, "_INDEXER_KERNEL", True),
            mock.patch.object(dsa_mod, "_INDEXER_KERNEL_PREFILL", True),
            mock.patch.object(dsa_mod, "_PREFILL_SPARSE", 1),
        ]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])

        setups = {flag: self._make_setup() for flag in (False, True)}
        toks = {flag: [] for flag in setups}

        def cache_loc_of(tok_lists, s_lens):
            out = []
            for t, sl in zip(tok_lists, s_lens):
                out.extend(list(t) + [0] * (-(-sl // ps) * ps - len(t)))
            return out

        def step(prefix_lens, ext_lens):
            seq_lens = [p + e for p, e in zip(prefix_lens, ext_lens)]
            n = sum(ext_lens)
            x = {
                "q": rng.standard_normal((n, self.num_heads, self.kv_lora_rank), np.float32) * 0.3,
                "q_pe": rng.standard_normal((n, self.num_heads, self.rope), np.float32) * 0.3,
                "k": rng.standard_normal((n, self.kv_lora_rank), np.float32) * 0.3,
                "k_pe": rng.standard_normal((n, self.rope), np.float32) * 0.3,
                "q_idx": rng.standard_normal((n, self.idx_heads, self.idx_dim), np.float32),
                "k_idx": rng.standard_normal((n, self.idx_dim), np.float32),
                "w": rng.random((n, self.idx_heads), np.float32),
            }
            positions = [p + i for p, e in zip(prefix_lens, ext_lens) for i in range(e)]
            outs = {}
            for flag, (pool, alloc, backend, layer) in setups.items():
                t = toks[flag]
                if not t:
                    t.extend([] for _ in seq_lens)
                last = [lst[-1] if lst else -1 for lst in t]
                loc = alloc.alloc_extend(prefix_lens, seq_lens, last, n)
                cur = 0
                for i, e in enumerate(ext_lens):
                    t[i].extend(int(v) for v in loc[cur : cur + e])
                    cur += e
                outs[flag] = self._run(pool, backend, layer, seq_lens, ext_lens,
                                       cache_loc_of(t, seq_lens), loc, positions, x, flag)
            self.assertGreater(float(np.abs(outs[False]).max()), 0.0)
            np.testing.assert_allclose(outs[True], outs[False], rtol=1e-5, atol=1e-5,
                                       err_msg=f"prefix {prefix_lens} extend {ext_lens}")
            return seq_lens

        # Four sequences with multi-page contexts; then mixed steps whose one-token rows sit
        # before, between and after the prefill chunks.
        s = step([0, 0, 0, 0], [71, 45, 25, 93])
        s = step(s, [1, 7, 1, 1])
        s = step(s, [1, 1, 11, 1])
        s = step(s, [6, 1, 1, 5])
        step(s, [1, 1, 1, 1])  # one-token rows only (d == n)


if __name__ == "__main__":
    unittest.main()
