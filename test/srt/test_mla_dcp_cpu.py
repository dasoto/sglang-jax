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
from jax.sharding import NamedSharding
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

    def setUp(self):
        # Both the dcp_size=1 baseline and the DCP path run the kernel's CPU reference.
        env = mock.patch.dict(os.environ, {"SGLANG_JAX_MLA_CPU_REFERENCE": "1"})
        env.start()
        self.addCleanup(env.stop)

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
        q_pe_dev = jax.device_put(
            q_pe, NamedSharding(self.mesh, P("data", "tensor", None))
        )
        k_c_dev = jax.device_put(k_c, NamedSharding(self.mesh, P("data", None)))
        k_pe_dev = jax.device_put(k_pe, NamedSharding(self.mesh, P("data", None)))

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
            )
        pool.replace_buffer([updated_cache])
        return np.asarray(jax.device_get(o_latent))

    def test_extend_and_decode_parity(self):
        rng = np.random.default_rng(20261021)
        page_size = 16
        num_heads = 4
        kv_lora_rank = 128
        qk_rope_head_dim = 64

        pool1, alloc1, backend1, layer1 = self._make_setup(
            dcp_size=1, page_size=page_size
        )
        pool2, alloc2, backend2, layer2 = self._make_setup(
            dcp_size=2, page_size=page_size
        )

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
            pages = [
                toks[m * page_size] // page_size
                for m in range((len(toks) + page_size - 1) // page_size)
            ]
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

        q = (
            rng.standard_normal((total_q, num_heads, kv_lora_rank), dtype=np.float32)
            * 0.2
        )
        q_pe = (
            rng.standard_normal(
                (total_q, num_heads, qk_rope_head_dim), dtype=np.float32
            )
            * 0.2
        )
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
            q_d = (
                rng.standard_normal((bs, num_heads, kv_lora_rank), dtype=np.float32)
                * 0.2
            )
            q_pe_d = (
                rng.standard_normal((bs, num_heads, qk_rope_head_dim), dtype=np.float32)
                * 0.2
            )
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

        q_e = (
            rng.standard_normal((total_ext, num_heads, kv_lora_rank), dtype=np.float32)
            * 0.2
        )
        q_pe_e = (
            rng.standard_normal(
                (total_ext, num_heads, qk_rope_head_dim), dtype=np.float32
            )
            * 0.2
        )
        k_c_e = rng.standard_normal((total_ext, kv_lora_rank), dtype=np.float32) * 0.2
        k_pe_e = (
            rng.standard_normal((total_ext, qk_rope_head_dim), dtype=np.float32) * 0.2
        )

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


if __name__ == "__main__":
    unittest.main()


class TestDCPPageOwnership(unittest.TestCase):
    """Page j of every sequence must live on DCP rank j % dcp_size; the DSA
    backend derives each rank's local pages from that rule."""

    def _allocator(self, dcp_size: int, num_pages: int, page_size: int = 4):
        pool = SimpleNamespace(dcp_size=dcp_size)
        return PagedTokenToKVPoolAllocator(
            size=num_pages * page_size, page_size=page_size, kvcache=pool, dp_size=1
        )

    def test_extend_and_decode_pages_follow_round_robin_owner(self):
        page_size, dcp = 4, 2
        alloc = self._allocator(dcp, num_pages=32, page_size=page_size)
        seq_lens = [9, 4, 13]
        out = alloc.alloc_extend([0, 0, 0], seq_lens, [-1, -1, -1], sum(seq_lens))
        self.assertIsNotNone(out)
        starts = np.cumsum([0] + seq_lens)
        for i, n in enumerate(seq_lens):
            pages = out[starts[i] : starts[i + 1] : page_size] // page_size
            np.testing.assert_array_equal(
                (pages - 1) % dcp, np.arange(len(pages)) % dcp
            )
        # Decode: seq 1 (4 tokens, page 0 full) grows into page 1 -> rank 1.
        last_loc = [out[starts[i + 1] - 1] for i in range(3)]
        dec = alloc.alloc_decode([10, 5, 14], last_loc)
        self.assertEqual((dec[1] // page_size - 1) % dcp, 1)
        self.assertEqual(dec[0], last_loc[0] + 1)  # still inside its page 2

    def test_available_size_tracks_scarcest_rank(self):
        page_size, dcp = 4, 2
        alloc = self._allocator(dcp, num_pages=8, page_size=page_size)
        self.assertEqual(alloc.available_size(), 8 * page_size)
        # Single-page sequences all take rank-0 pages (pages 1, 3, 5, 7).
        for _ in range(3):
            self.assertIsNotNone(alloc.alloc_extend([0], [page_size], [-1], page_size))
        self.assertEqual(alloc.available_size(), 1 * dcp * page_size)
        # accounting (leak check, usage) still sees every free page
        self.assertEqual(alloc.free_size(), 5 * page_size)
        self.assertIsNotNone(alloc.alloc_extend([0], [page_size], [-1], page_size))
        self.assertEqual(alloc.available_size(), 0)
        # Rank 1 still has 4 pages, but a new sequence's page 0 must be on rank 0.
        free_before = alloc.free_pages[0].copy()
        self.assertIsNone(alloc.alloc_extend([0], [page_size], [-1], page_size))
        np.testing.assert_array_equal(alloc.free_pages[0], free_before)
