"""CPU correctness tests for MLA Decode Context Parallelism (DCP).

Verifies numerical parity between dcp_size=1 and dcp_size=2 across:
1. Multi-sequence EXTEND (including sequences with 0 owned pages on some DCP ranks,
   and sequences spanning multiple pages across DCP ranks).
2. Prefix-cached EXTEND (prefix_len > 0, extend_seq_len < seq_len).
3. Multi-step DECODE (including crossing a page boundary mid-decode).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionBackend
from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardMode
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


class TestMLADCPCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        devices = jax.devices("cpu")
        assert len(devices) >= 2, f"Expected >= 2 CPU devices, got {len(devices)}"
        cls.mesh = create_device_mesh(
            ici_parallelism=[1, 2], dcn_parallelism=[1, 1], devices=devices[:2]
        )

    def _make_setup(self, dcp_size: int, page_size: int = 16, total_tokens: int = 256):
        num_heads = 4
        kv_lora_rank = 128
        qk_nope_head_dim = 64
        qk_rope_head_dim = 64
        v_head_dim = 64

        pool = MLATokenToKVPool(
            size=total_tokens,
            page_size=page_size,
            dtype=jnp.float32,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            layer_num=1,
            mesh=self.mesh,
            dp_size=1,
            dcp_size=dcp_size,
        )
        allocator = PagedTokenToKVPoolAllocator(
            size=total_tokens,
            page_size=page_size,
            kvcache=pool,
            dp_size=1,
        )
        backend = MLAAttentionBackend(
            num_attn_heads=num_heads,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            page_size=page_size,
            mesh=self.mesh,
            dcp_size=dcp_size,
            num_kv_pages_per_block=(2, 1, 2),
            num_queries_per_block=(1, 1, 8),
            decode_batch_size=1,
        )
        layer = SimpleNamespace(
            layer_id=0,
            scaling=1.0 / np.sqrt(qk_nope_head_dim + qk_rope_head_dim),
            sliding_window_size=None,
            logit_cap=None,
        )
        return pool, allocator, backend, layer

    def _run_step(
        self,
        pool,
        backend,
        layer,
        forward_mode,
        seq_lens,
        extend_seq_lens,
        cache_loc,
        out_cache_loc,
        q,
        k_c,
        q_pe,
        k_pe,
        absorb_weights=None,
    ):
        worker_batch = SimpleNamespace(
            dp_size=1,
            per_dp_bs_size=len(seq_lens),
            seq_lens=np.array(seq_lens, dtype=np.int32),
            extend_seq_lens=(
                np.array(extend_seq_lens, dtype=np.int32)
                if extend_seq_lens is not None
                else None
            ),
            cache_loc=np.array(cache_loc, dtype=np.int32),
            out_cache_loc=np.array(out_cache_loc, dtype=np.int32),
            forward_mode=forward_mode,
        )
        backend.forward_metadata = backend.get_forward_metadata(worker_batch)
        forward_batch = SimpleNamespace(
            out_cache_loc=jax.device_put(
                np.array(out_cache_loc, dtype=np.int32),
                NamedSharding(self.mesh, P("data")),
            )
        )
        q_dev = jax.device_put(q, NamedSharding(self.mesh, P("data", "tensor", None)))
        q_pe_dev = jax.device_put(q_pe, NamedSharding(self.mesh, P("data", "tensor", None)))
        k_c_dev = jax.device_put(k_c, NamedSharding(self.mesh, P("data", None)))
        k_pe_dev = jax.device_put(k_pe, NamedSharding(self.mesh, P("data", None)))
        extra = {}
        if absorb_weights is not None:
            rep = NamedSharding(self.mesh, P(None, None, None))
            extra["absorb_weights"] = tuple(jax.device_put(w, rep) for w in absorb_weights)

        with jax.set_mesh(self.mesh):
            o_latent, updated_cache = backend(
                q_dev,
                k_c_dev,
                k_c_dev,
                layer,
                forward_batch,
                pool,
                q_rope=q_pe_dev,
                k_rope=k_pe_dev,
                **extra,
            )
        pool.replace_buffer([updated_cache])
        return np.asarray(jax.device_get(o_latent))

    def test_dcp_size_through_hybrid_wrapper(self):
        """Hybrid (KDA + MLA) models wrap the MLA backend; the absorb path must still see its DCP size."""
        from sgl_jax.srt.models.deepseek_v3 import _dcp_size

        mla = SimpleNamespace(dcp_size=32)
        self.assertEqual(_dcp_size(mla), 32)
        self.assertEqual(_dcp_size(SimpleNamespace(full_attn_backend=mla)), 32)
        self.assertEqual(_dcp_size(SimpleNamespace()), 1)
        self.assertEqual(_dcp_size(lambda *a, **k: None), 1)

    def test_replicated_absorb_matches_latent_path(self):
        """absorb_weights: W_UK / W_UV folded in around the DCP collectives, both combines."""
        for rs_combine in ("0", "1"):
            with mock.patch.dict(os.environ, {"SGLANG_JAX_MLA_DCP_RS_COMBINE": rs_combine}):
                self._check_replicated_absorb(rs_combine)

    def test_rs_combine_bf16_close_to_f32(self):
        """SGLANG_JAX_MLA_DCP_RS_BF16: the weighted partial outputs reduce-scatter in bf16."""
        page_size, num_heads, kv_lora_rank, d_k, d_v, rope = 16, 4, 128, 64, 64, 64
        seq_lens = [35, 16, 20]
        outs = {}
        for bf16 in ("0", "1"):
            env = {"SGLANG_JAX_MLA_DCP_RS_COMBINE": "1", "SGLANG_JAX_MLA_DCP_RS_BF16": bf16}
            with mock.patch.dict(os.environ, env):
                rng = np.random.default_rng(11)
                pool, alloc, backend, layer = self._make_setup(dcp_size=2, page_size=page_size)
                w_uk = rng.standard_normal((kv_lora_rank, num_heads, d_k), dtype=np.float32) * 0.1
                w_uv = rng.standard_normal((kv_lora_rank, num_heads, d_v), dtype=np.float32) * 0.1
                n = sum(seq_lens)
                q_nope = rng.standard_normal((n, num_heads, d_k), dtype=np.float32) * 0.5
                q_pe = rng.standard_normal((n, num_heads, rope), dtype=np.float32) * 0.2
                k_c = rng.standard_normal((n, kv_lora_rank), dtype=np.float32) * 0.2
                k_pe = rng.standard_normal((n, rope), dtype=np.float32) * 0.2
                out_loc = alloc.alloc_extend([0, 0, 0], seq_lens, [-1, -1, -1], n)
                toks, cur = [], 0
                for sl in seq_lens:
                    t = list(out_loc[cur : cur + sl])
                    aligned = ((sl + page_size - 1) // page_size) * page_size
                    toks.extend(t + [0] * (aligned - sl))
                    cur += sl
                cache_loc = np.array(toks, dtype=np.int32)
                outs[bf16] = self._run_step(
                    pool, backend, layer, ForwardMode.EXTEND, seq_lens, seq_lens, cache_loc,
                    out_loc, q_nope, k_c, q_pe, k_pe, absorb_weights=(w_uk, w_uv),
                )
        ref, out = outs["0"], outs["1"]
        scale = float(np.abs(ref).max())
        self.assertGreater(scale, 0.0)
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-2 * scale)
        self.assertLess(float(np.abs(out - ref).mean()), 3e-3 * float(np.abs(ref).mean()))

    def _check_replicated_absorb(self, rs_combine):
        rng = np.random.default_rng(7)
        page_size, num_heads, kv_lora_rank, d_k, d_v, rope = 16, 4, 128, 64, 64, 64
        pool_a, alloc_a, backend_a, layer_a = self._make_setup(dcp_size=2, page_size=page_size)
        pool_b, alloc_b, backend_b, layer_b = self._make_setup(dcp_size=2, page_size=page_size)
        w_uk = rng.standard_normal((kv_lora_rank, num_heads, d_k), dtype=np.float32) * 0.1
        w_uv = rng.standard_normal((kv_lora_rank, num_heads, d_v), dtype=np.float32) * 0.1

        def both(mode, seq_lens, ext_lens, cache_loc, out_loc, n):
            q_nope = rng.standard_normal((n, num_heads, d_k), dtype=np.float32) * 0.5
            q_pe = rng.standard_normal((n, num_heads, rope), dtype=np.float32) * 0.2
            k_c = rng.standard_normal((n, kv_lora_rank), dtype=np.float32) * 0.2
            k_pe = rng.standard_normal((n, rope), dtype=np.float32) * 0.2
            ql = np.einsum("thd,rhd->thr", q_nope, w_uk)
            o_lat = self._run_step(
                pool_a, backend_a, layer_a, mode, seq_lens, ext_lens, cache_loc, out_loc,
                ql, k_c, q_pe, k_pe,
            )
            ref = np.einsum("thr,rhd->thd", o_lat, w_uv)
            out = self._run_step(
                pool_b, backend_b, layer_b, mode, seq_lens, ext_lens, cache_loc, out_loc,
                q_nope, k_c, q_pe, k_pe, absorb_weights=(w_uk, w_uv),
            )
            self.assertEqual(out.shape, ref.shape)
            np.testing.assert_allclose(
                out, ref, rtol=1e-4, atol=1e-4, err_msg=f"{mode} rs_combine={rs_combine}"
            )

        seq_lens = [35, 16, 20]
        out_loc = alloc_a.alloc_extend([0, 0, 0], seq_lens, [-1, -1, -1], sum(seq_lens))
        self.assertTrue(
            np.array_equal(
                out_loc, alloc_b.alloc_extend([0, 0, 0], seq_lens, [-1, -1, -1], sum(seq_lens))
            )
        )
        req_tokens, cursor = [], 0
        for s_len in seq_lens:
            req_tokens.append(list(out_loc[cursor : cursor + s_len]))
            cursor += s_len

        def cache_loc_of(toks_list, lens):
            chunks = []
            for toks, sl in zip(toks_list, lens):
                aligned = ((sl + page_size - 1) // page_size) * page_size
                chunks.extend(toks + [0] * (aligned - len(toks)))
            return np.array(chunks, dtype=np.int32)

        both(ForwardMode.EXTEND, seq_lens, seq_lens, cache_loc_of(req_tokens, seq_lens), out_loc,
             sum(seq_lens))
        for _ in range(2):
            last = [toks[-1] for toks in req_tokens]
            seq_lens = [s + 1 for s in seq_lens]
            dec = alloc_a.alloc_decode(seq_lens, last)
            self.assertTrue(np.array_equal(dec, alloc_b.alloc_decode(seq_lens, last)))
            for i in range(len(seq_lens)):
                req_tokens[i].append(int(dec[i]))
            both(ForwardMode.DECODE, seq_lens, None, cache_loc_of(req_tokens, seq_lens), dec,
                 len(seq_lens))

    def test_extend_and_decode_parity(self):
        rng = np.random.default_rng(20261021)
        page_size = 16
        num_heads = 4
        kv_lora_rank = 128
        qk_rope_head_dim = 64

        pool1, alloc1, backend1, layer1 = self._make_setup(dcp_size=1, page_size=page_size)
        pool2, alloc2, backend2, layer2 = self._make_setup(dcp_size=2, page_size=page_size)

        # Verify per-device cache shape is halved under dcp_size=2
        local_pages_1 = pool1.get_fused_kv_buffer(0).addressable_shards[0].data.shape[0]
        local_pages_2 = pool2.get_fused_kv_buffer(0).addressable_shards[0].data.shape[0]
        self.assertEqual((local_pages_1 - 1) // 2 + 1, local_pages_2)

        # 3 sequences:
        # - seq 0: len 35 (3 pages: 2 on rank 0, 1 on rank 1)
        # - seq 1: len 16 (1 exact page: 1 on rank 0, 0 on rank 1 -> exercises dummy page path!)
        # - seq 2: len 20 (2 pages: 1 on rank 0, 1 on rank 1)
        seq_lens = [35, 16, 20]
        prefix_lens = [0, 0, 0]
        last_locs = [-1, -1, -1]
        total_q = sum(seq_lens)

        out_loc_1 = alloc1.alloc_extend(prefix_lens, seq_lens, last_locs, total_q)
        out_loc_2 = alloc2.alloc_extend(prefix_lens, seq_lens, last_locs, total_q)

        # Build padded page-aligned cache_loc per sequence for each allocator
        req_tokens_1 = []
        req_tokens_2 = []
        cursor = 0
        for s_len in seq_lens:
            req_tokens_1.append(list(out_loc_1[cursor : cursor + s_len]))
            req_tokens_2.append(list(out_loc_2[cursor : cursor + s_len]))
            cursor += s_len

        # Verify alloc2 stripes pages of each sequence across DCP ranks 0, 1, 0, ...
        for toks in req_tokens_2:
            pages = [toks[m * page_size] // page_size for m in range((len(toks) + page_size - 1) // page_size)]
            for m, p in enumerate(pages):
                self.assertEqual((p - 1) % 2, m % 2)

        def build_cache_loc(req_toks, s_lens):
            chunks = []
            for toks, sl in zip(req_toks, s_lens):
                aligned = ((sl + page_size - 1) // page_size) * page_size
                padded = toks + [0] * (aligned - len(toks))
                chunks.extend(padded)
            return np.array(chunks, dtype=np.int32)

        cache_loc_1 = build_cache_loc(req_tokens_1, seq_lens)
        cache_loc_2 = build_cache_loc(req_tokens_2, seq_lens)

        q = rng.standard_normal((total_q, num_heads, kv_lora_rank), dtype=np.float32) * 0.2
        q_pe = rng.standard_normal((total_q, num_heads, qk_rope_head_dim), dtype=np.float32) * 0.2
        k_c = rng.standard_normal((total_q, kv_lora_rank), dtype=np.float32) * 0.2
        k_pe = rng.standard_normal((total_q, qk_rope_head_dim), dtype=np.float32) * 0.2

        out_extend_1 = self._run_step(
            pool1,
            backend1,
            layer1,
            ForwardMode.EXTEND,
            seq_lens,
            seq_lens,
            cache_loc_1,
            out_loc_1,
            q,
            k_c,
            q_pe,
            k_pe,
        )
        out_extend_2 = self._run_step(
            pool2,
            backend2,
            layer2,
            ForwardMode.EXTEND,
            seq_lens,
            seq_lens,
            cache_loc_2,
            out_loc_2,
            q,
            k_c,
            q_pe,
            k_pe,
        )
        np.testing.assert_allclose(out_extend_2, out_extend_1, rtol=1e-4, atol=1e-4)

        # Now run 3 DECODE steps (seq 1 crosses page boundary 16 -> 17 on step 0!)
        for step in range(3):
            last_locs_1 = [toks[-1] for toks in req_tokens_1]
            last_locs_2 = [toks[-1] for toks in req_tokens_2]
            seq_lens = [s + 1 for s in seq_lens]
            dec_loc_1 = alloc1.alloc_decode(seq_lens, last_locs_1)
            dec_loc_2 = alloc2.alloc_decode(seq_lens, last_locs_2)
            for i in range(len(seq_lens)):
                req_tokens_1[i].append(int(dec_loc_1[i]))
                req_tokens_2[i].append(int(dec_loc_2[i]))
            cache_loc_1 = build_cache_loc(req_tokens_1, seq_lens)
            cache_loc_2 = build_cache_loc(req_tokens_2, seq_lens)

            bs = len(seq_lens)
            q_d = rng.standard_normal((bs, num_heads, kv_lora_rank), dtype=np.float32) * 0.2
            q_pe_d = rng.standard_normal((bs, num_heads, qk_rope_head_dim), dtype=np.float32) * 0.2
            k_c_d = rng.standard_normal((bs, kv_lora_rank), dtype=np.float32) * 0.2
            k_pe_d = rng.standard_normal((bs, qk_rope_head_dim), dtype=np.float32) * 0.2

            out_dec_1 = self._run_step(
                pool1,
                backend1,
                layer1,
                ForwardMode.DECODE,
                seq_lens,
                None,
                cache_loc_1,
                dec_loc_1,
                q_d,
                k_c_d,
                q_pe_d,
                k_pe_d,
            )
            out_dec_2 = self._run_step(
                pool2,
                backend2,
                layer2,
                ForwardMode.DECODE,
                seq_lens,
                None,
                cache_loc_2,
                dec_loc_2,
                q_d,
                k_c_d,
                q_pe_d,
                k_pe_d,
            )
            np.testing.assert_allclose(
                out_dec_2,
                out_dec_1,
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Mismatch at decode step {step}",
            )

        # Now test prefix-cached EXTEND: extend each sequence by 10 tokens!
        prefix_lens = list(seq_lens)
        extend_lens = [10, 10, 10]
        seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
        last_locs_1 = [toks[-1] for toks in req_tokens_1]
        last_locs_2 = [toks[-1] for toks in req_tokens_2]
        total_ext = sum(extend_lens)

        ext_loc_1 = alloc1.alloc_extend(prefix_lens, seq_lens, last_locs_1, total_ext)
        ext_loc_2 = alloc2.alloc_extend(prefix_lens, seq_lens, last_locs_2, total_ext)

        cursor = 0
        for i, e_len in enumerate(extend_lens):
            req_tokens_1[i].extend(list(ext_loc_1[cursor : cursor + e_len]))
            req_tokens_2[i].extend(list(ext_loc_2[cursor : cursor + e_len]))
            cursor += e_len
        cache_loc_1 = build_cache_loc(req_tokens_1, seq_lens)
        cache_loc_2 = build_cache_loc(req_tokens_2, seq_lens)

        q_e = rng.standard_normal((total_ext, num_heads, kv_lora_rank), dtype=np.float32) * 0.2
        q_pe_e = rng.standard_normal((total_ext, num_heads, qk_rope_head_dim), dtype=np.float32) * 0.2
        k_c_e = rng.standard_normal((total_ext, kv_lora_rank), dtype=np.float32) * 0.2
        k_pe_e = rng.standard_normal((total_ext, qk_rope_head_dim), dtype=np.float32) * 0.2

        out_pext_1 = self._run_step(
            pool1,
            backend1,
            layer1,
            ForwardMode.EXTEND,
            seq_lens,
            extend_lens,
            cache_loc_1,
            ext_loc_1,
            q_e,
            k_c_e,
            q_pe_e,
            k_pe_e,
        )
        out_pext_2 = self._run_step(
            pool2,
            backend2,
            layer2,
            ForwardMode.EXTEND,
            seq_lens,
            extend_lens,
            cache_loc_2,
            ext_loc_2,
            q_e,
            k_c_e,
            q_pe_e,
            k_pe_e,
        )
        np.testing.assert_allclose(out_pext_2, out_pext_1, rtol=1e-4, atol=1e-4)

    def test_decode_rows_first_matches_mixed_order(self):
        """SGLANG_JAX_MLA_DECODE_ROWS_FIRST: one-token rows of an EXTEND batch take the decode
        branches (distribution (d, d, n)) with the same outputs as the all-mixed layout, for
        one-token rows between and after the prefill chunks, with and without DCP."""
        page_size = 16
        num_heads, kv_lora_rank, qk_rope_head_dim = 4, 128, 64
        env = "SGLANG_JAX_MLA_DECODE_ROWS_FIRST"
        for dcp_size in (1, 2):
            rng = np.random.default_rng(20261029 + dcp_size)
            setups = {
                flag: self._make_setup(dcp_size=dcp_size, page_size=page_size)
                for flag in ("0", "1")
            }
            req_tokens = {flag: [] for flag in setups}

            def cache_loc_of(toks_list, s_lens):
                chunks = []
                for toks, sl in zip(toks_list, s_lens):
                    aligned = ((sl + page_size - 1) // page_size) * page_size
                    chunks.extend(list(toks) + [0] * (aligned - len(toks)))
                return np.array(chunks, dtype=np.int32)

            def step(prefix_lens, extend_lens, check_order):
                seq_lens = [p + e for p, e in zip(prefix_lens, extend_lens)]
                total = sum(extend_lens)
                q = rng.standard_normal((total, num_heads, kv_lora_rank), dtype=np.float32) * 0.2
                q_pe = (
                    rng.standard_normal((total, num_heads, qk_rope_head_dim), dtype=np.float32)
                    * 0.2
                )
                k_c = rng.standard_normal((total, kv_lora_rank), dtype=np.float32) * 0.2
                k_pe = rng.standard_normal((total, qk_rope_head_dim), dtype=np.float32) * 0.2
                outs = {}
                for flag, (pool, alloc, backend, layer) in setups.items():
                    toks = req_tokens[flag]
                    last = [t[-1] if t else -1 for t in toks] or [-1] * len(seq_lens)
                    if not toks:
                        toks.extend([] for _ in seq_lens)
                        last = [-1] * len(seq_lens)
                    loc = alloc.alloc_extend(prefix_lens, seq_lens, last, total)
                    cursor = 0
                    for i, e in enumerate(extend_lens):
                        toks[i].extend(int(x) for x in loc[cursor : cursor + e])
                        cursor += e
                    with mock.patch.dict(os.environ, {env: flag if check_order else "0"}):
                        outs[flag] = self._run_step(
                            pool,
                            backend,
                            layer,
                            ForwardMode.EXTEND,
                            seq_lens,
                            extend_lens,
                            cache_loc_of(toks, seq_lens),
                            loc,
                            q,
                            k_c,
                            q_pe,
                            k_pe,
                        )
                    if check_order and flag == "1":
                        n_one = sum(1 for e in extend_lens if e == 1)
                        dist = np.asarray(jax.device_get(backend.forward_metadata.distribution))
                        self.assertEqual(list(dist), [n_one, n_one, len(seq_lens)])
                np.testing.assert_allclose(
                    outs["1"],
                    outs["0"],
                    rtol=1e-4,
                    atol=1e-4,
                    err_msg=f"dcp={dcp_size} extend={extend_lens}",
                )
                return seq_lens

            # Prefill four sequences (same layout in both setups), then two mixed steps.
            seq_lens = step([0, 0, 0, 0], [35, 16, 20, 24], check_order=False)
            seq_lens = step(seq_lens, [10, 1, 1, 7], check_order=True)  # rows between chunks
            step(seq_lens, [5, 3, 1, 1], check_order=True)  # rows after the chunks

if __name__ == "__main__":
    unittest.main()
