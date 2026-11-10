"""DP page migration: plan layout and the cross-group copy on a 2 x 2 CPU mesh (data x tensor, DCP 2).

Run with:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python -m pytest python/sgl_jax/test/mem_cache/test_dp_migrate.py -v
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from sgl_jax.srt.mem_cache.dp_migrate import DpPageMigrator, plan_copy  # noqa: E402

DP, DCP, LOCAL_PAGES = 2, 2, 6  # 6 chip-local pages per chip; group-local page ids 1..10


def test_plan_copy_groups_pages_by_owner_and_pads():
    src = np.array([1, 2, 3, 4, 5])  # owners 0 1 0 1 0
    dst = np.array([7, 4, 9, 2, 3])  # owners 0 1 0 1 0
    s, d, v = plan_copy(src, dst, dcp=2, bucket=4)
    assert s.shape == d.shape == v.shape == (2, 4)
    # rank 0: pages 1, 3, 5 -> local 1, 2, 3; destinations 7, 9, 3 -> local 4, 5, 2
    np.testing.assert_array_equal(s[0], [1, 2, 3, 0])
    np.testing.assert_array_equal(d[0], [4, 5, 2, 0])
    np.testing.assert_array_equal(v[0], [True, True, True, False])
    # rank 1: pages 2, 4 -> local 1, 2; destinations 4, 2 -> local 2, 1
    np.testing.assert_array_equal(s[1], [1, 2, 0, 0])
    np.testing.assert_array_equal(d[1], [2, 1, 0, 0])


def test_plan_copy_rejects_owner_mismatch_and_small_bucket():
    with pytest.raises(ValueError, match="same DCP owner"):
        plan_copy(np.array([1]), np.array([2]), dcp=2)
    with pytest.raises(ValueError, match="bucket"):
        plan_copy(np.array([1, 3, 5]), np.array([1, 3, 5]), dcp=2, bucket=2)


class _Pool:
    """Stand-in for MLATokenToKVPool: per-layer latent buffers plus indexer buffers."""

    def __init__(self, kv, idx):
        self.kv_buffer = kv
        self.indexer_key_buffer = idx

    def replace_buffer(self, bufs):
        kv, idx = bufs
        self.kv_buffer = list(kv)
        self.indexer_key_buffer[: len(idx)] = idx


def _global_row(dp_rank, page):
    """Row of group ``dp_rank``'s page ``page`` in the [DP * DCP * LOCAL_PAGES, ...] buffer."""
    rank = (page - 1) % DCP
    return (dp_rank * DCP + rank) * LOCAL_PAGES + (page - 1) // DCP + 1


@pytest.mark.parametrize("buffers_per_call", [1, 8])
def test_copy_moves_pages_between_groups_in_every_buffer(buffers_per_call):
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(DP, DCP), ("data", "tensor"))
    spec = P(("data", "tensor"), None, None)
    sharding = NamedSharding(mesh, spec)
    rows = DP * DCP * LOCAL_PAGES
    rng = np.random.default_rng(0)
    host = [
        rng.integers(-100, 100, size=(rows, 3, 4)).astype(np.int8),
        rng.integers(-100, 100, size=(rows, 3, 4)).astype(np.int8),
        rng.standard_normal((rows, 2, 8)).astype(jnp.bfloat16),
    ]
    pool = _Pool(
        [jax.device_put(host[0], sharding), jax.device_put(host[1], sharding)],
        [jax.device_put(host[2], sharding)],
    )
    migrator = DpPageMigrator(mesh, [pool], dcp=DCP, buffers_per_call=buffers_per_call)
    src = np.array([1, 2, 3, 4, 5])  # group 0, sequence pages 0..4
    dst = np.array([7, 4, 9, 2, 3])  # group 1, same owners
    assert migrator.copy(0, 1, src, dst) == 5

    out = [np.asarray(b) for b in pool.kv_buffer + pool.indexer_key_buffer]
    for h, o in zip(host, out):
        expected = h.copy()
        for s, d in zip(src, dst):
            expected[_global_row(1, d)] = h[_global_row(0, s)]
        np.testing.assert_array_equal(o, expected)
