# cd python && USE_DEVICE_TYPE=cpu python -m pytest sgl_jax/test/mem_cache/test_dcp_alloc_retry.py -v
"""retry_dcp_alloc: under DCP an allocation can fail on one short owner rank while
most of the pool is evictable cache; it evicts more (doubling) and retries."""

import unittest
from types import SimpleNamespace

from sgl_jax.srt.mem_cache.common import retry_dcp_alloc


class _Cache:
    def __init__(self, dcp_size, evictable):
        self.token_to_kv_pool_allocator = SimpleNamespace(
            dcp_size=dcp_size, backup_state=lambda: ("backup", len(self.evicted))
        )
        self._evictable = evictable
        self.evicted = []

    def evictable_size(self, dp_rank=0):
        return self._evictable

    def evict(self, params):
        n = min(params.num_tokens, self._evictable)
        self._evictable -= n
        self.evicted.append((params.num_tokens, params.dp_rank))


class TestRetryDcpAlloc(unittest.TestCase):
    def test_evicts_more_until_the_short_rank_fits(self):
        cache = _Cache(dcp_size=32, evictable=10**6)
        attempts = []

        def alloc():  # succeeds once 3 more eviction rounds freed the short rank
            attempts.append(len(cache.evicted))
            return "slots" if len(cache.evicted) >= 3 else None

        out, state = retry_dcp_alloc(cache, 1000, 2, alloc, backup_state=True)
        self.assertEqual(out, "slots")
        self.assertEqual([n for n, _ in cache.evicted], [2000, 4000, 8000])
        self.assertEqual({r for _, r in cache.evicted}, {2})
        self.assertEqual(state, ("backup", 3))  # backup taken right before the success

    def test_stops_when_nothing_is_evictable(self):
        cache = _Cache(dcp_size=32, evictable=3000)
        out, state = retry_dcp_alloc(cache, 1000, 0, lambda: None, backup_state=False)
        self.assertIsNone(out)
        self.assertEqual([n for n, _ in cache.evicted], [2000, 4000])  # 3000 tokens gone
        self.assertIsNone(state)

    def test_no_retry_without_dcp(self):
        cache = _Cache(dcp_size=1, evictable=10**6)
        out, _ = retry_dcp_alloc(cache, 1000, 0, lambda: "slots", backup_state=False)
        self.assertIsNone(out)
        self.assertEqual(cache.evicted, [])


if __name__ == "__main__":
    unittest.main()
