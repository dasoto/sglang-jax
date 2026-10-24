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
    def test_unary_chain_keeps_only_tail_snapshots(self):
        cache, req_pool, allocator = _build_cache(page_size=128, interval=128)
        tokens = list(range(1, 769))  # 6 chunks of 128
        req = _new_req("r1", tokens, cache)
        self.assertEqual(len(req.fill_ids), 0)

        for start in range(0, 640, 128):
            _alloc_and_step_chunk(
                cache, req_pool, allocator, req, tokens[start : start + 128], is_finished=False
            )
            cache.assert_recurrent_slot_ledger(0, [req])

        _alloc_and_step_chunk(cache, req_pool, allocator, req, tokens[640:768], is_finished=True)
        cache.assert_recurrent_slot_ledger(0, [])
        # A unary chain of 6 chunks keeps only the last two prompt boundaries
        # (640 and the 768 leaf), not one snapshot per chunk.
        self.assertEqual(cache.recurrent_evictable_size(0), 2)

    def test_unary_chain_legacy_keeps_only_leaf_snapshot(self):
        old = os.environ.get("SGLANG_JAX_RECURRENT_TAIL_CHECKPOINTS")
        os.environ["SGLANG_JAX_RECURRENT_TAIL_CHECKPOINTS"] = "0"
        try:
            cache, req_pool, allocator = _build_cache(page_size=128, interval=128)
        finally:
            if old is None:
                os.environ.pop("SGLANG_JAX_RECURRENT_TAIL_CHECKPOINTS")
            else:
                os.environ["SGLANG_JAX_RECURRENT_TAIL_CHECKPOINTS"] = old
        tokens = list(range(1, 385))  # 3 chunks of 128
        req = _new_req("r1", tokens, cache)
        _alloc_and_step_chunk(cache, req_pool, allocator, req, tokens[0:128], is_finished=False)
        _alloc_and_step_chunk(cache, req_pool, allocator, req, tokens[128:256], is_finished=False)
        _alloc_and_step_chunk(cache, req_pool, allocator, req, tokens[256:384], is_finished=True)
        cache.assert_recurrent_slot_ledger(0, [])
        self.assertEqual(cache.recurrent_evictable_size(0), 1)

    def _run_turn_with_decode(self, cache, req_pool, allocator, rid, prompt, decode):
        """Chunked prefill of ``prompt`` (128-token chunks), then a decode run."""
        req = _new_req(rid, prompt, cache)
        start = len(req.fill_ids)
        while start < len(prompt):
            end = min(start + 128 - start % 128, len(prompt))
            _alloc_and_step_chunk(
                cache, req_pool, allocator, req, prompt[start:end], is_finished=False
            )
            start = end
        req.output_ids = list(decode)
        _alloc_and_step_chunk(cache, req_pool, allocator, req, decode, is_finished=True)
        cache.assert_recurrent_slot_ledger(0, [])
        return req

    def test_prompt_end_checkpoint_survives_decode_extension(self):
        # Turn N decodes past a track boundary; turn N+1 resends turn N's prompt
        # followed by a different assistant message (agentic replay). Before the
        # prompt-tail checkpoints the prompt-end snapshot was freed once the
        # decode tokens became its only child, so turn N+1 missed entirely.
        cache, req_pool, allocator = _build_cache(page_size=128, interval=128)
        prompt = list(range(1, 385))  # 384 tokens
        decode = list(range(5000, 5128))  # crosses the 512 boundary
        self._run_turn_with_decode(cache, req_pool, allocator, "t0", prompt, decode)

        next_turn = prompt + list(range(7000, 7200))
        r1 = _new_req("t1", next_turn, cache)
        self.assertEqual(len(r1.prefix_indices), 384)
        self.assertIsNotNone(r1.recurrent_cow_src_index)
        cache.dec_lock_ref(r1.last_node, r1.cache_lock_params)

    def test_prompt_tail_checkpoint_covers_pullback(self):
        # Turn N+1 drops the last tokens of turn N's prompt (a trace pull-back),
        # diverging before the prompt-end boundary: the previous boundary hits.
        cache, req_pool, allocator = _build_cache(page_size=128, interval=128)
        prompt = list(range(1, 385))  # 384 tokens: boundaries 128, 256, 384
        decode = list(range(5000, 5256))  # ends on the 640 boundary
        self._run_turn_with_decode(cache, req_pool, allocator, "t0", prompt, decode)

        pullback = prompt[:300] + list(range(8000, 8100))
        r1 = _new_req("t1", pullback, cache)
        self.assertEqual(len(r1.prefix_indices), 256)
        cache.dec_lock_ref(r1.last_node, r1.cache_lock_params)

        # Prompt-tail snapshots stay LRU-evictable: nothing is pinned.
        evictable = cache.recurrent_evictable_size(0)
        evict_res = cache.evict(EvictParams(recurrent_num=evictable, dp_rank=0))
        self.assertEqual(evict_res.recurrent_num_evicted, evictable)
        self.assertEqual(cache.recurrent_evictable_size(0), 0)
        cache.assert_recurrent_slot_ledger(0, [])

    def test_sequential_shared_prefix_branch_retains_snapshot(self):
        cache, req_pool, allocator = _build_cache(page_size=128, interval=128)
        prefix_p = list(range(100, 228))  # 128 tokens
        suffix_a = list(range(1000, 1128))  # 128 tokens
        suffix_b = list(range(2000, 2128))  # 128 tokens
        suffix_c = list(range(3000, 3128))  # 128 tokens

        # R1 = [P, A] runs and finishes at the end of its prefill: the leaf [P, A]
        # and the previous prompt boundary N_P (a prompt-tail checkpoint) keep
        # snapshots.
        r1_tokens = prefix_p + suffix_a
        r1 = _new_req("r1", r1_tokens, cache)
        _alloc_and_step_chunk(cache, req_pool, allocator, r1, prefix_p, is_finished=False)
        _alloc_and_step_chunk(cache, req_pool, allocator, r1, suffix_a, is_finished=True)
        cache.assert_recurrent_slot_ledger(0, [])
        self.assertEqual(cache.recurrent_evictable_size(0), 2)

        # R2 = [P, B] hits N_P and finishes at [P, B]: N_P (now a branch point),
        # N_PA and N_PB hold snapshots (3 total).
        r2_tokens = prefix_p + suffix_b
        r2 = _new_req("r2", r2_tokens, cache)
        self.assertEqual(len(r2.prefix_indices), 128)
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

    def test_commit_on_internal_node_retains_branch_snapshot(self):
        cache, req_pool, allocator = _build_cache(page_size=128, interval=128)
        prefix_p = list(range(100, 228))  # 128 tokens
        tail_a = list(range(1000, 1384))  # 384 tokens
        suffix_b = list(range(2000, 2128))
        suffix_c = list(range(3000, 3128))

        # R1 = [P, A] (4 chunks): N_P is outside the prompt tail, so only the
        # 384 boundary and the 512 leaf keep snapshots.
        r1 = _new_req("r1", prefix_p + tail_a, cache)
        _alloc_and_step_chunk(cache, req_pool, allocator, r1, prefix_p, is_finished=False)
        _alloc_and_step_chunk(cache, req_pool, allocator, r1, tail_a[:128], is_finished=False)
        _alloc_and_step_chunk(cache, req_pool, allocator, r1, tail_a[128:256], is_finished=False)
        _alloc_and_step_chunk(cache, req_pool, allocator, r1, tail_a[256:], is_finished=True)
        cache.assert_recurrent_slot_ledger(0, [])
        self.assertEqual(cache.recurrent_evictable_size(0), 2)

        # R2 = [P, B] misses, chunks at P and commits its snapshot on the
        # existing internal node N_P, which is retained as a branch point.
        r2 = _new_req("r2", prefix_p + suffix_b, cache)
        self.assertEqual(len(r2.prefix_indices), 0)
        _alloc_and_step_chunk(cache, req_pool, allocator, r2, prefix_p, is_finished=False)
        cache.assert_recurrent_slot_ledger(0, [r2])
        _alloc_and_step_chunk(cache, req_pool, allocator, r2, suffix_b, is_finished=True)
        cache.assert_recurrent_slot_ledger(0, [])
        self.assertEqual(cache.recurrent_evictable_size(0), 4)

        # R3 = [P, C] hits N_P.
        r3 = _new_req("r3", prefix_p + suffix_c, cache)
        self.assertEqual(len(r3.prefix_indices), 128)
        self.assertIsNotNone(r3.recurrent_cow_src_index)
        _alloc_and_step_chunk(cache, req_pool, allocator, r3, suffix_c, is_finished=True)
        cache.assert_recurrent_slot_ledger(0, [])
        self.assertEqual(cache.recurrent_evictable_size(0), 5)

        evict_res = cache.evict(EvictParams(recurrent_num=5, dp_rank=0))
        self.assertEqual(evict_res.recurrent_num_evicted, 5)
        self.assertEqual(cache.recurrent_evictable_size(0), 0)
        cache.assert_recurrent_slot_ledger(0, [])


class _EnvOverride:
    def __init__(self, **env):
        self.env = env
        self.old = {}

    def __enter__(self):
        for k, v in self.env.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestRecurrentSnapshotRetention(unittest.TestCase):
    """Run F: decode-leaf-first eviction and queued-prefix pinning."""

    def _turn(self, cache, req_pool, allocator, rid, prompt, decode):
        return TestRecurrentBranchPrefixCache._run_turn_with_decode(
            self, cache, req_pool, allocator, rid, prompt, decode
        )

    def _two_sessions(self, decode_first: str):
        with _EnvOverride(SGLANG_JAX_RECURRENT_EVICT_DECODE_LEAF_FIRST=decode_first,
                          SGLANG_JAX_RECURRENT_TAIL_CHECKPOINTS="1"):
            cache, req_pool, allocator = _build_cache(page_size=128, interval=128)
        prompt_a = list(range(1, 385))  # 384 tokens; prompt-end checkpoint at 384
        self._turn(cache, req_pool, allocator, "a0", prompt_a, list(range(5000, 5128)))  # leaf 512
        prompt_b = list(range(20001, 20385))
        self._turn(cache, req_pool, allocator, "b0", prompt_b, list(range(6000, 6128)))
        return cache, req_pool, allocator, prompt_a

    def test_lru_evicts_prompt_end_checkpoint_first_by_default(self):
        cache, _, _, prompt_a = self._two_sessions("0")
        # Session A's prompt-end checkpoint is its oldest snapshot: plain LRU drops it.
        cache.evict(EvictParams(recurrent_num=1, dp_rank=0))
        r1 = _new_req("a1", prompt_a + list(range(7000, 7200)), cache)
        self.assertEqual(len(r1.prefix_indices), 0)
        cache.dec_lock_ref(r1.last_node, r1.cache_lock_params)

    def test_decode_leaf_evicted_before_prompt_end_checkpoint(self):
        cache, _, _, prompt_a = self._two_sessions("1")
        # Both decode leaves (past the prompt ends) go before any prompt-end checkpoint.
        res = cache.evict(EvictParams(recurrent_num=2, dp_rank=0))
        self.assertEqual(res.recurrent_num_evicted, 2)
        r1 = _new_req("a1", prompt_a + list(range(7000, 7200)), cache)
        self.assertEqual(len(r1.prefix_indices), 384)
        self.assertIsNotNone(r1.recurrent_cow_src_index)
        cache.dec_lock_ref(r1.last_node, r1.cache_lock_params)
        cache.assert_recurrent_slot_ledger(0, [])

    def test_queued_pin_protects_prefix_until_released(self):
        from types import SimpleNamespace

        from sgl_jax.srt.managers import scheduler as sched

        cache, req_pool, allocator, prompt_a = self._two_sessions("0")
        old_min = sched._PIN_QUEUED_MIN_TOKENS
        sched._PIN_QUEUED_MIN_TOKENS = 128
        try:
            fake = SimpleNamespace(tree_cache=cache)
            queued = Req(
                rid="a1",
                origin_input_text="",
                origin_input_ids=prompt_a + list(range(7000, 7200)),
                sampling_params=SamplingParams(max_new_tokens=1),
                dp_rank=0,
                eos_token_ids={999999},
                vocab_size=32000,
            )
            sched.Scheduler._pin_queued_prefix(fake, queued)
            self.assertIsNotNone(getattr(queued, "_queue_pin", None))
            # Evict everything evictable: the pinned prompt-end checkpoint survives.
            cache.evict(EvictParams(recurrent_num=cache.recurrent_evictable_size(0), dp_rank=0))
            r1 = _new_req("a1-admit", prompt_a + list(range(7000, 7200)), cache)
            self.assertEqual(len(r1.prefix_indices), 384)
            # Admission: the request holds its own lock, then the queue pin is released.
            sched.Scheduler._release_queue_pin(fake, queued)
            self.assertIsNone(queued._queue_pin)
            cache.dec_lock_ref(r1.last_node, r1.cache_lock_params)
            cache.assert_recurrent_slot_ledger(0, [])
            self.assertGreater(cache.recurrent_evictable_size(0), 0)
            # Releasing twice is a no-op.
            sched.Scheduler._release_queue_pin(fake, queued)
        finally:
            sched._PIN_QUEUED_MIN_TOKENS = old_min

    def test_short_prefix_is_not_pinned(self):
        from types import SimpleNamespace

        from sgl_jax.srt.managers import scheduler as sched

        cache, _, _, prompt_a = self._two_sessions("0")
        fake = SimpleNamespace(tree_cache=cache)
        queued = Req(
            rid="a1",
            origin_input_text="",
            origin_input_ids=prompt_a + list(range(7000, 7200)),
            sampling_params=SamplingParams(max_new_tokens=1),
            dp_rank=0,
            eos_token_ids={999999},
            vocab_size=32000,
        )
        sched.Scheduler._pin_queued_prefix(fake, queued)  # default minimum is 4096 tokens
        self.assertIsNone(getattr(queued, "_queue_pin", None))


if __name__ == "__main__":
    unittest.main()
