"""DP rebalancing in the scheduler: a waiting request moves to another DP group with the part of its cached prefix
that group lacks (page copy, radix insert, lock until admission), and the next request of an overloaded group
moves to the least-loaded one. Uses a real EAGLE radix cache and DCP allocator; the page copy is recorded."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from sgl_jax.srt.managers import scheduler as sched_mod
from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.managers.scheduler import Scheduler
from sgl_jax.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sgl_jax.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sgl_jax.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sgl_jax.srt.sampling.sampling_params import SamplingParams

PS, DP, DCP = 4, 2, 2


class _Migrator:
    dcp = DCP

    def __init__(self):
        self.copies = []

    def copy(self, src_dp, dst_dp, src_pages, dst_pages):
        self.copies.append((src_dp, dst_dp, np.asarray(src_pages), np.asarray(dst_pages)))
        return len(src_pages)


class TestSchedulerDpRebalance(unittest.TestCase):
    def setUp(self):
        self.alloc = PagedTokenToKVPoolAllocator(
            size=DP * 64 * PS, page_size=PS, kvcache=SimpleNamespace(dcp_size=DCP), dp_size=DP
        )
        self.tree = RadixCache(None, self.alloc, page_size=PS, is_eagle=True)
        s = Scheduler.__new__(Scheduler)
        s.tree_cache, s.token_to_kv_pool_allocator, s.page_size = self.tree, self.alloc, PS
        s.dp_size, s.per_dp_max_running_requests = DP, 16
        s.dp_page_migrator = _Migrator()
        s._dp_rebalance_stats = [0, 0, 0, 0.0]
        s._prefill_sched_step = 0
        s.waiting_queue = []
        s.chunked_reqs = [None] * DP
        s._preempted_chunked = MethodType(lambda self: [None] * DP, s)
        s.running_batch = SimpleNamespace(
            reqs_info=[SimpleNamespace(reqs=[], batch_is_full=False) for _ in range(DP)]
        )
        self.s = s
        self.prompt = list(range(100, 100 + 6 * PS + 1))  # 25 tokens -> 24 bigram slots, 6 pages

    def _req(self, ids, dp_rank, rid="r"):
        req = Req(
            rid=rid,
            origin_input_text="",
            origin_input_ids=list(ids),
            sampling_params=SamplingParams(max_new_tokens=8),
            dp_rank=dp_rank,
            eos_token_ids={2},
            vocab_size=1000,
        )
        req._dp_pinned = False
        return req

    def _cache(self, tokens, dp_rank):
        """Insert ``tokens`` (bigram slots = len - 1) for ``dp_rank``; returns the slots."""
        slots = self.alloc.alloc((len(tokens) - 1) // PS * PS, dp_rank=dp_rank)
        self.tree.insert(InsertParams(key=RadixKey(list(tokens), None, dp_rank), value=slots))
        return slots

    def _match(self, req, dp_rank):
        key = req.match_key()
        key = RadixKey(key.token_ids, key.extra_key, dp_rank)
        return self.tree.match_prefix(MatchPrefixParams(key=key, req=req))

    def test_migrate_copies_prefix_and_locks_it_until_released(self):
        src_slots = self._cache(self.prompt, 0)
        req = self._req(self.prompt + [7, 8], 0)
        self.assertTrue(self.s._dp_migrate_waiting_req(req, 0, 1))

        self.assertEqual(req.dp_rank, 1)
        ((src_dp, dst_dp, src_pages, dst_pages),) = self.s.dp_page_migrator.copies
        self.assertEqual((src_dp, dst_dp), (0, 1))
        np.testing.assert_array_equal(src_pages, src_slots[::PS] // PS)
        # Sequence page j stays on DCP rank j % DCP.
        np.testing.assert_array_equal((dst_pages - 1) % DCP, np.arange(6) % DCP)
        got = self._match(req, 1)
        np.testing.assert_array_equal(
            got.device_indices, (dst_pages[:, None] * PS + np.arange(PS)).reshape(-1)
        )
        self.assertEqual(self.tree.protected_size(1), 6 * PS)
        self.assertEqual(self.tree.evictable_size(1), 0)
        self.s._dp_release_migrate_lock(req)
        self.assertIsNone(req._dp_migrate_lock)
        self.assertEqual(self.tree.protected_size(1), 0)
        self.assertEqual(self.tree.evictable_size(1), 6 * PS)
        self.assertEqual(self.s._dp_rebalance_stats[:3], [1, 6, 0])

    def test_migrate_copies_only_pages_the_destination_lacks(self):
        self._cache(self.prompt, 0)
        have = self._cache(self.prompt[: 2 * PS + 1], 1)  # destination holds the first 2 pages
        req = self._req(self.prompt + [7, 8], 0)
        self.assertTrue(self.s._dp_migrate_waiting_req(req, 0, 1))

        ((_, _, src_pages, dst_pages),) = self.s.dp_page_migrator.copies
        self.assertEqual(len(src_pages), 4)
        np.testing.assert_array_equal((dst_pages - 1) % DCP, np.arange(2, 6) % DCP)
        got = self._match(req, 1).device_indices
        np.testing.assert_array_equal(got[: 2 * PS], have)
        self.assertEqual(len(got), 6 * PS)

    def test_copy_keeps_the_destination_reserve(self):
        self._cache(self.prompt, 0)
        req = self._req(self.prompt + [7, 8], 0)
        # Fill rank 1 so that copying 6 pages would leave less than the reserve.
        free = self.alloc.available_size(1)
        self.alloc.alloc(free - 8 * PS, dp_rank=1)
        with patch.object(sched_mod, "_DP_REBALANCE_RESERVE", 0.1):
            self.assertFalse(self.s._dp_migrate_waiting_req(req, 0, 1))
        self.assertEqual(req.dp_rank, 0)
        self.assertEqual(self.s.dp_page_migrator.copies, [])
        with patch.object(sched_mod, "_DP_REBALANCE_RESERVE", 0.0):
            self.assertTrue(self.s._dp_migrate_waiting_req(req, 0, 1))

    def test_uncached_request_moves_without_a_copy(self):
        req = self._req(self.prompt, 0)
        self.assertTrue(self.s._dp_migrate_waiting_req(req, 0, 1))
        self.assertEqual(req.dp_rank, 1)
        self.assertEqual(self.s.dp_page_migrator.copies, [])
        self.assertIsNone(getattr(req, "_dp_migrate_lock", None))
        self.assertEqual(self.s._dp_rebalance_stats[2], 1)

    def test_rebalance_moves_only_the_overloaded_ranks_next_request(self):
        self._cache(self.prompt, 0)
        head = self._req(self.prompt + [7, 8], 0, "head")
        second = self._req(self.prompt + [9, 10], 0, "second")
        other = self._req(self.prompt, 1, "other")
        self.s.waiting_queue = [head, second, other]
        with (
            patch.object(sched_mod, "_DP_REBALANCE_GAP", 2),
            patch.object(sched_mod, "_DP_REBALANCE_MAX", 2),
            patch.object(sched_mod, "_DP_REBALANCE_FORCE", False),
        ):
            self.s._dp_rebalance_waiting([5, 1])
        self.assertEqual([r.dp_rank for r in self.s.waiting_queue], [1, 0, 1])

    def test_rebalance_moves_several_requests_per_rank_when_allowed(self):
        self._cache(self.prompt, 0)
        reqs = [self._req(self.prompt + [7, k], 0, f"r{k}") for k in range(3)]
        self.s.waiting_queue = list(reqs)
        with (
            patch.object(sched_mod, "_DP_REBALANCE_GAP", 2),
            patch.object(sched_mod, "_DP_REBALANCE_MAX", 6),
            patch.object(sched_mod, "_DP_REBALANCE_PER_RANK", 2),
            patch.object(sched_mod, "_DP_REBALANCE_FORCE", False),
        ):
            self.s._dp_rebalance_waiting([6, 0])
        self.assertEqual([r.dp_rank for r in reqs], [1, 1, 0])
        # The second move found the prefix already copied: no second page copy.
        self.assertEqual(len(self.s.dp_page_migrator.copies), 1)

    def test_busy_destination_only_when_allowed(self):
        self._cache(self.prompt, 0)
        head = self._req(self.prompt + [7, 8], 0, "head")
        self.s.waiting_queue = [head]
        self.s.chunked_reqs = [None, self._req(self.prompt, 1, "chunk")]
        with (
            patch.object(sched_mod, "_DP_REBALANCE_GAP", 2),
            patch.object(sched_mod, "_DP_REBALANCE_FORCE", False),
        ):
            with patch.object(sched_mod, "_DP_REBALANCE_BUSY_DEST", False):
                self.s._dp_rebalance_waiting([6, 0])
                self.assertEqual(head.dp_rank, 0)
            with patch.object(sched_mod, "_DP_REBALANCE_BUSY_DEST", True):
                self.s._dp_rebalance_waiting([6, 0])
                self.assertEqual(head.dp_rank, 1)

    def test_rebalance_respects_gap_and_pins(self):
        self._cache(self.prompt, 0)
        head = self._req(self.prompt + [7, 8], 0, "head")
        self.s.waiting_queue = [head]
        with (
            patch.object(sched_mod, "_DP_REBALANCE_GAP", 2),
            patch.object(sched_mod, "_DP_REBALANCE_FORCE", False),
        ):
            self.s._dp_rebalance_waiting([3, 2])  # gap 1 < 2
            self.assertEqual(head.dp_rank, 0)
            head._dp_pinned = True
            self.s._dp_rebalance_waiting([9, 0])
            self.assertEqual(head.dp_rank, 0)
        self.assertEqual(self.s.dp_page_migrator.copies, [])


if __name__ == "__main__":
    unittest.main()
