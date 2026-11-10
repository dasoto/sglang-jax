"""Copy KV pages between the DP groups of one paged MLA pool (session migration).

A group-local page id ``g >= 1`` lives on DCP rank ``(g - 1) % dcp`` at chip-local page ``(g - 1) // dcp + 1``,
and page ``j`` of a sequence must sit on rank ``j % dcp``, so a migrated page keeps its DCP rank and the copy
runs down each DCP column of the mesh: the source group's chip gathers its pages, a psum over the ``data`` axis
broadcasts them along the column (every other group contributes zeros), and the destination group's chip writes
them. Pages per rank are padded to a few buckets so each bucket compiles once.
"""

from __future__ import annotations

import contextlib
import logging
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

logger = logging.getLogger(__name__)

PAGE_BUCKETS = (8, 16, 32, 64, 128)
# Buffers (layers) copied per program. A program's temporaries are one allocation and serving HBM is fragmented
# (largest free block ~35-50 MB under AgentX), so keep it small: 8 buffers take ~100 MB at 128 pages per rank.
_BUFFERS_PER_CALL = int(os.environ.get("SGLANG_JAX_DP_MIGRATE_BUFFERS", "2"))


def plan_copy(
    src_pages: np.ndarray, dst_pages: np.ndarray, dcp: int, bucket: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chip-local ``[dcp, n]`` source and destination page indices for group-local page pairs.

    ``src_pages[k]`` and ``dst_pages[k]`` must have the same DCP owner. Row ``r`` lists rank ``r``'s pairs in
    order; padding entries point at local page 0 with ``valid`` False (the kernel rewrites page 0 unchanged).
    """
    src = np.asarray(src_pages, np.int64)
    dst = np.asarray(dst_pages, np.int64)
    if src.shape != dst.shape or src.ndim != 1:
        raise ValueError(f"page lists must be equal 1-D arrays, got {src.shape} and {dst.shape}")
    if len(src) and (src.min() < 1 or dst.min() < 1):
        raise ValueError("page ids start at 1")
    owner = (src - 1) % dcp
    if not np.array_equal(owner, (dst - 1) % dcp):
        raise ValueError("source and destination pages must have the same DCP owner")
    counts = np.bincount(owner, minlength=dcp) if len(src) else np.zeros(dcp, np.int64)
    n = int(counts.max()) if bucket is None else bucket
    if n < int(counts.max()):
        raise ValueError(f"bucket {n} < {int(counts.max())} pages on one rank")
    src_local = np.zeros((dcp, n), np.int32)
    dst_local = np.zeros((dcp, n), np.int32)
    valid = np.zeros((dcp, n), bool)
    for r in range(dcp):
        sel = owner == r
        k = int(sel.sum())
        src_local[r, :k] = (src[sel] - 1) // dcp + 1
        dst_local[r, :k] = (dst[sel] - 1) // dcp + 1
        valid[r, :k] = True
    return src_local, dst_local, valid


def _bucket_for(n: int) -> int | None:
    for b in PAGE_BUCKETS:
        if n <= b:
            return b
    return None


def _copy_body(src_dp, dst_dp, src_local, dst_local, valid, *bufs):
    d = jax.lax.axis_index("data")
    # One row per DCP rank; without DCP the pages are replicated over ``tensor`` and every rank copies row 0.
    r = jax.lax.axis_index("tensor") % src_local.shape[0]
    s, t, v = src_local[r], dst_local[r], valid[r]
    is_src, is_dst = d == src_dp, d == dst_dp
    outs = []
    for b in bufs:
        moved = b[s]
        moved = jax.lax.psum(jnp.where(is_src, moved, jnp.zeros_like(moved)), "data")
        keep = (is_dst & v).reshape((-1,) + (1,) * (b.ndim - 1))
        outs.append(b.at[t].set(jnp.where(keep, moved, b[t])))
    return tuple(outs)


class DpPageMigrator:
    """Copies a sequence's pages from one DP group to another in every pool (target and draft)."""

    def __init__(self, mesh, pools, dcp: int, buffers_per_call: int = _BUFFERS_PER_CALL):
        self.mesh = mesh
        self.pools = [p for p in pools if p is not None]
        self.dcp = dcp
        self.buffers_per_call = max(1, buffers_per_call)
        self._fns: dict = {}

    def _buffers(self, pool) -> tuple[list, list]:
        return list(pool.kv_buffer), list(getattr(pool, "indexer_key_buffer", None) or [])

    def _fn(self, bufs: list):
        key = tuple((b.shape, str(b.dtype), b.sharding.spec) for b in bufs)
        fn = self._fns.get(key)
        if fn is None:
            k = len(bufs)
            specs = tuple(b.sharding.spec for b in bufs)
            mapped = jax.shard_map(
                _copy_body,
                mesh=self.mesh,
                in_specs=(P(), P(), P(), P(), P()) + specs,
                out_specs=specs,
                check_vma=False,
            )
            fn = jax.jit(mapped, donate_argnums=tuple(range(5, 5 + k)))
            self._fns[key] = fn
        return fn

    def _run(self, src_dp: int, dst_dp: int, src_local, dst_local, valid) -> None:
        args = (
            jnp.int32(src_dp),
            jnp.int32(dst_dp),
            jnp.asarray(src_local),
            jnp.asarray(dst_local),
            jnp.asarray(valid),
        )
        for pool in self.pools:
            lock = getattr(pool, "_donate_lock", None)
            with lock if lock is not None else contextlib.nullcontext():
                kv, idx = self._buffers(pool)
                bufs = kv + idx
                out = []
                for i in range(0, len(bufs), self.buffers_per_call):
                    chunk = bufs[i : i + self.buffers_per_call]
                    out.extend(self._fn(chunk)(*args, *chunk))
                pool.replace_buffer((out[: len(kv)], out[len(kv) :]))

    def copy(self, src_dp: int, dst_dp: int, src_pages: np.ndarray, dst_pages: np.ndarray) -> int:
        """Copy group ``src_dp``'s pages ``src_pages`` to group ``dst_dp``'s pages ``dst_pages``.

        Dispatches asynchronously; device order puts it before any later program. Returns pages copied.
        """
        src_pages = np.asarray(src_pages, np.int64)
        dst_pages = np.asarray(dst_pages, np.int64)
        if src_dp == dst_dp or len(src_pages) == 0:
            return 0
        owner = (src_pages - 1) % self.dcp
        per_rank = int(np.bincount(owner, minlength=self.dcp).max())
        step = PAGE_BUCKETS[-1] * self.dcp
        if per_rank > PAGE_BUCKETS[-1]:
            # Split by sequence position: consecutive pages cycle through the ranks.
            for i in range(0, len(src_pages), step):
                self.copy(src_dp, dst_dp, src_pages[i : i + step], dst_pages[i : i + step])
            return len(src_pages)
        bucket = _bucket_for(per_rank)
        src_local, dst_local, valid = plan_copy(src_pages, dst_pages, self.dcp, bucket)
        self._run(src_dp, dst_dp, src_local, dst_local, valid)
        return len(src_pages)

    def precompile(self) -> None:
        """Compile every bucket (a no-op copy: no valid pages)."""
        for b in PAGE_BUCKETS:
            z = np.zeros((self.dcp, b), np.int32)
            self._run(0, 0, z, z, np.zeros((self.dcp, b), bool))
        for pool in self.pools:
            kv, _ = self._buffers(pool)
            jax.block_until_ready(kv[0])
        try:
            bufs = sum(self._buffers(self.pools[0]), [])[: self.buffers_per_call]
            z = np.zeros((self.dcp, PAGE_BUCKETS[-1]), np.int32)
            lowered = self._fn(bufs).lower(
                jnp.int32(0), jnp.int32(0), z, z, np.zeros_like(z, bool), *bufs
            )
            mem = lowered.compile().memory_analysis()
            logger.info(
                "DP page migrator: %d buckets x %d pools compiled; %d buffers per call, temp %.1f MB "
                "at %d pages per rank",
                len(PAGE_BUCKETS),
                len(self.pools),
                self.buffers_per_call,
                getattr(mem, "temp_size_in_bytes", 0) / 2**20,
                PAGE_BUCKETS[-1],
            )
        except Exception as e:  # noqa: BLE001
            logger.info("DP page migrator compiled (memory analysis unavailable: %s)", e)
