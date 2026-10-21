"""Unit tests for RecurrentComponent branch-node snapshot retention and eviction
in UnifiedRadixCache."""

from __future__ import annotations

import os
import unittest

if os.environ.get("USE_DEVICE_TYPE") == "cpu" or "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import EvictParams, MatchPrefixParams
from sgl_jax.srt.mem_cache.cache_init_params import CacheInitParams
from sgl_jax.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    MHATokenToKVPool,
)
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool
from sgl_jax.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sgl_jax.srt.mem_cache.unified_radix_cache import RadixKey, UnifiedRadixCache
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

_DEVICES = jax.devices()[:2] if len(jax.devices()) >= 2 else jax.devices()
_MESH = create_device_mesh(
    ici_parallelism=[1, len(_DEVICES)], dcn_parallelism=[1, 1], devices=_DEVICES
)


def _build_cache(page_size: int = 128, interval: int = 128, num_slots: int = 16):
    kv_pool = MHATokenToKVPool(
        size=page_size * 64,
        page_size=page_size,
        dtype=jnp.bfloat16,
        head_num=2,
        head_dim=128,
        layer_num=1,
        mesh=_MESH,
        dp_size=1,
        abstract=True,
    )
    allocator = PagedTokenToKVPoolAllocator(
        size=page_size * 64,
        page_size=page_size,
        kvcache=kv_pool,
        dp_size=1,
    )
    rsp = RecurrentStatePool(
        linear_recurrent_layer_ids=[0],
        size=num_slots,
        num_heads=2,
        head_dim=16,
        conv_kernel_size=4,
        mesh=_MESH,
        dp_size=1,
        temporal_dtype=jnp.float32,
        conv_dtype=jnp.bfloat16,
    )
    req_pool = HybridReqToTokenPool(
        size=8,
        max_context_len=2048,
        dtype=np.int32,
        recurrent_state_pool=rsp,
        dp_size=1,
        enable_recurrent_extra_buffer=True,
        ping_pong_slots=2,
    )
    cache = UnifiedRadixCache(
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
        disable=False,
        tree_components=(ComponentType.FULL, ComponentType.RECURRENT),
        enable_recurrent_extra_buffer=True,
        recurrent_track_interval=interval,
    )
    return cache, req_pool, allocator


def _alloc_and_step_chunk(
    cache: UnifiedRadixCache,
    req_pool: HybridReqToTokenPool,
    allocator: PagedTokenToKVPoolAllocator,
    req: Req,
    chunk_tokens: list[int],
    *,
    is_finished: bool,
):
    prev_len = len(req.fill_ids)
    req.fill_ids = list(req.fill_ids) + list(chunk_tokens)
    new_len = len(req.fill_ids)
    if req.req_pool_idx is None:
        slots = req_pool.alloc([req])
        assert slots is not None
        if prev_len > 0:
            req_pool.write((req.req_pool_idx, slice(0, prev_len)), req.prefix_indices)

    kv_locs = allocator.alloc(len(chunk_tokens), dp_rank=0)
    assert kv_locs is not None
    req_pool.write((req.req_pool_idx, slice(prev_len, new_len)), kv_locs)
    req.kv_committed_len = new_len

    # Simulate track boundary hit at new_len
    if new_len % cache.recurrent_track_interval == 0:
        req.recurrent_last_track_seqlen = new_len
        req.recurrent_next_track_idx = req_pool.get_recurrent_ping_pong_other_idx(
            req.recurrent_next_track_idx
        )

    if is_finished:
        cache.cache_finished_req(req)
        req_pool.free(req)
    else:
        cache.cache_unfinished_req(req)


def _new_req(rid: str, token_ids: list[int], cache: UnifiedRadixCache) -> Req:
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=list(token_ids),
        sampling_params=SamplingParams(max_new_tokens=1),
        dp_rank=0,
        eos_token_ids={999999},
        vocab_size=32000,
    )
    req.fill_ids = []
    match = cache.match_prefix(
        MatchPrefixParams(
            key=RadixKey(token_ids=list(token_ids), extra_key=None, dp_rank=0),
            req=req,
            cow_recurrent=True,
        )
    )
    matched_len = len(match.device_indices)
    req.prefix_indices = match.device_indices
    req.last_node = match.last_device_node
    req.last_matched_prefix_len = matched_len
    req.cache_protected_len = matched_len
    lock_res = cache.inc_lock_ref(req.last_node)
    req.cache_lock_params = lock_res.to_dec_params()
    req.fill_ids = list(token_ids[:matched_len])
    return req


class TestRecurrentBranchPrefixCache(unittest.TestCase):
    def test_unary_chain_keeps_only_leaf_snapshot(self):
        cache, req_pool, allocator = _build_cache(page_size=128, interval=128)
        tokens = list(range(1, 385))  # 3 chunks of 128
        req = _new_req("r1", tokens, cache)
        self.assertEqual(len(req.fill_ids), 0)

        _alloc_and_step_chunk(cache, req_pool, allocator, req, tokens[0:128], is_finished=False)
        cache.assert_recurrent_slot_ledger(0, [req])

        _alloc_and_step_chunk(cache, req_pool, allocator, req, tokens[128:256], is_finished=False)
        cache.assert_recurrent_slot_ledger(0, [req])

        _alloc_and_step_chunk(cache, req_pool, allocator, req, tokens[256:384], is_finished=True)
        cache.assert_recurrent_slot_ledger(0, [])
        # Unary chain of 3 chunks must hold only 1 recurrent snapshot (at the leaf).
        self.assertEqual(cache.recurrent_evictable_size(0), 1)

    def test_sequential_shared_prefix_branch_retains_snapshot(self):
        cache, req_pool, allocator = _build_cache(page_size=128, interval=128)
        prefix_p = list(range(100, 228))  # 128 tokens
        suffix_a = list(range(1000, 1128))  # 128 tokens
        suffix_b = list(range(2000, 2128))  # 128 tokens
        suffix_c = list(range(3000, 3128))  # 128 tokens

        # R1 = [P, A] runs and finishes -> unary chain, leaf at [P, A].
        r1_tokens = prefix_p + suffix_a
        r1 = _new_req("r1", r1_tokens, cache)
        _alloc_and_step_chunk(cache, req_pool, allocator, r1, prefix_p, is_finished=False)
        _alloc_and_step_chunk(cache, req_pool, allocator, r1, suffix_a, is_finished=True)
        cache.assert_recurrent_slot_ledger(0, [])
        self.assertEqual(cache.recurrent_evictable_size(0), 1)

        # R2 = [P, B] arrives, chunks at P (128), attaches snapshot to internal node N_P,
        # and then finishes at [P, B]. Both N_P and N_PB + N_PA hold snapshots (3 total).
        r2_tokens = prefix_p + suffix_b
        r2 = _new_req("r2", r2_tokens, cache)
        _alloc_and_step_chunk(cache, req_pool, allocator, r2, prefix_p, is_finished=False)
        cache.assert_recurrent_slot_ledger(0, [r2])
        _alloc_and_step_chunk(cache, req_pool, allocator, r2, suffix_b, is_finished=True)
        cache.assert_recurrent_slot_ledger(0, [])
        self.assertEqual(cache.recurrent_evictable_size(0), 3)

        # R3 = [P, C] arrives -> immediate prefix cache hit on P (128 tokens) + CoW src slot!
        r3_tokens = prefix_p + suffix_c
        r3 = _new_req("r3", r3_tokens, cache)
        self.assertEqual(len(r3.prefix_indices), 128)
        self.assertIsNotNone(r3.recurrent_cow_src_index)
        self.assertGreater(r3.recurrent_cow_src_index, 0)
        _alloc_and_step_chunk(cache, req_pool, allocator, r3, suffix_c, is_finished=True)
        cache.assert_recurrent_slot_ledger(0, [])
        self.assertEqual(cache.recurrent_evictable_size(0), 4)

        # Evict all 4 recurrent snapshots (3 leaves + 1 internal branch node) and verify ledger.
        evict_res = cache.evict(EvictParams(recurrent_num=4, dp_rank=0))
        self.assertEqual(evict_res.recurrent_num_evicted, 4)
        self.assertEqual(cache.recurrent_evictable_size(0), 0)
        cache.assert_recurrent_slot_ledger(0, [])


if __name__ == "__main__":
    unittest.main()
