"""Superseded prompt-tail checkpoints (SGLANG_JAX_RECURRENT_EVICT_SUPERSEDED_FIRST=1).

A session's earlier turns leave retained checkpoints on its own path; once a later turn
finishes, the ones more than prompt_tail_window tokens above its prompt end are marked
superseded and evicted before other sessions' snapshots, regardless of LRU order. Branch
points (>= 2 children) are never marked.

run: cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_recurrent_superseded_eviction.py
"""
import os

if os.environ.get("USE_DEVICE_TYPE") == "cpu":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    os.environ["JAX_PLATFORMS"] = "cpu"

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool, MHATokenToKVPool
from sgl_jax.srt.mem_cache.radix_cache import RadixKey
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.mem_cache.unified_cache_components import ComponentType
from sgl_jax.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)


class TestRecurrentSupersededEviction(unittest.TestCase):
    def _cache(self, flag: str):
        os.environ["SGLANG_JAX_RECURRENT_EVICT_SUPERSEDED_FIRST"] = flag
        os.environ["SGLANG_JAX_RECURRENT_TAIL_CHECKPOINTS"] = "2"
        state_pool = RecurrentStatePool(linear_recurrent_layer_ids=[0, 1], size=16, num_heads=4, head_dim=8,
                                        conv_kernel_size=4, mesh=mesh, dp_size=1)
        pool = HybridReqToTokenPool(size=8, max_context_len=8192, dtype=np.int32, recurrent_state_pool=state_pool,
                                    dp_size=1, enable_recurrent_extra_buffer=True, ping_pong_slots=2)
        kv = MHATokenToKVPool(size=16384, page_size=1, dtype=jnp.bfloat16, head_num=1, head_dim=8, layer_num=1,
                              mesh=mesh, dp_size=1)
        alloc = TokenToKVPoolAllocator(size=16384, kvcache=kv, dp_size=1)
        cache = UnifiedRadixCache(req_to_token_pool=pool, token_to_kv_pool_allocator=alloc, page_size=1,
                                  disable=False, kv_head_num=1, head_dim=8, layer_num=1, max_seq_len=8192,
                                  dtype=jnp.bfloat16, tree_components=(ComponentType.FULL, ComponentType.RECURRENT),
                                  enable_recurrent_extra_buffer=True, recurrent_track_interval=128)
        return pool, cache, cache.components[ComponentType.RECURRENT]

    def _insert(self, pool, cache, ids, start_val):
        slot = pool.alloc_recurrent_slot(0)
        self.assertIsNotNone(slot)
        cache.insert(InsertParams(key=RadixKey(list(ids)), value=np.arange(start_val, start_val + len(ids), dtype=np.int32),
                                  recurrent_value=pool.recurrent_value_from_slot(slot)))
        node = cache.match_prefix(__import__("sgl_jax.srt.mem_cache.base_prefix_cache", fromlist=["x"]).MatchPrefixParams(
            key=RadixKey(list(ids)))).best_match_node
        return node

    def _build(self, flag):
        pool, cache, comp = self._cache(flag)
        rng = np.random.default_rng(0)
        a = [int(x) for x in rng.integers(1, 1000, 3000)]
        b = [int(x) for x in rng.integers(1000, 2000, 1500)]
        # session A: three turns, prompt ends at 1000 / 2000 / 3000 tokens, each retained
        n1 = self._insert(pool, cache, a[:1000], 1)
        comp._retain_checkpoint(n1)
        n2 = self._insert(pool, cache, a[:2000], 1)
        comp._retain_checkpoint(n2)
        n3 = self._insert(pool, cache, a[:3000], 1)
        # session B: one idle snapshot, older than everything on A's path in LRU order
        nb = self._insert(pool, cache, b, 5001)
        for n in (n1, n2, n3):
            n.last_access_time += 10 ** 9
        return pool, cache, comp, n1, n2, n3, nb

    def test_marks_older_unary_checkpoints(self):
        pool, cache, comp, n1, n2, n3, nb = self._build("1")
        self.assertEqual(len(n3.key) if n3.parent is n2 else -1, 1000)
        comp._mark_superseded_ancestors(n3)
        self.assertTrue(getattr(n2, "_recurrent_superseded", False))
        self.assertTrue(getattr(n1, "_recurrent_superseded", False))
        self.assertFalse(getattr(n3, "_recurrent_superseded", False))
        self.assertFalse(getattr(nb, "_recurrent_superseded", False))

    def test_superseded_evicted_before_idle_session(self):
        pool, cache, comp, n1, n2, n3, nb = self._build("1")
        comp._mark_superseded_ancestors(n3)
        ct = ComponentType.RECURRENT
        cache.evict(EvictParams(recurrent_num=2, dp_rank=0))
        self.assertIsNone(n1.component_data[ct].value)
        self.assertIsNone(n2.component_data[ct].value)
        self.assertIsNotNone(nb.component_data[ct].value)
        self.assertIsNotNone(n3.component_data[ct].value)

    def test_flag_off_keeps_lru_order(self):
        pool, cache, comp, n1, n2, n3, nb = self._build("0")
        comp._mark_superseded_ancestors(n3)  # marking alone changes nothing with the flag off
        ct = ComponentType.RECURRENT
        cache.evict(EvictParams(recurrent_num=1, dp_rank=0))
        self.assertIsNone(nb.component_data[ct].value)  # LRU victim: the idle session

    def test_branch_point_not_marked(self):
        pool, cache, comp, n1, n2, n3, nb = self._build("1")
        a2 = list(n2.key.token_ids)  # unused; build a fork below n1 so it has two children
        rng = np.random.default_rng(1)
        prefix = []
        node = n1
        while node is not None and node is not cache.root_node:
            prefix = list(node.key.token_ids) + prefix
            node = node.parent
        fork = prefix + [int(x) for x in rng.integers(3000, 4000, 300)]
        self._insert(pool, cache, fork, 9001)
        self.assertGreaterEqual(len(n1.children), 2)
        comp._mark_superseded_ancestors(n3)
        self.assertFalse(getattr(n1, "_recurrent_superseded", False))
        self.assertTrue(getattr(n2, "_recurrent_superseded", False))


if __name__ == "__main__":
    unittest.main()
