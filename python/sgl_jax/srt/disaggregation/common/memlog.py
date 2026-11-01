"""Periodic device-memory log for PD servers, to tell leaked buffers from piling-up
programs: every SGLANG_JAX_PD_MEMLOG_S seconds (0 = off) process 0 logs device 0's
memory stats, the live arrays resident on it grouped by (shape, dtype), and the
number of loaded executables."""

from __future__ import annotations

import collections
import logging
import math
import os
import time

import jax

logger = logging.getLogger(__name__)

_PERIOD_S = float(os.environ.get("SGLANG_JAX_PD_MEMLOG_S", "0"))
_last_log = 0.0
_prev_count: collections.Counter = collections.Counter()


def maybe_log_device_memory(tag: str) -> None:
    global _last_log, _prev_count
    if _PERIOD_S <= 0 or jax.process_index() != 0:
        return
    now = time.monotonic()
    if now - _last_log < _PERIOD_S:
        return
    _last_log = now
    try:
        import gc

        collected = gc.collect()  # unreachable cycles holding device arrays show up here
        dev = jax.local_devices()[0]
        stats = dev.memory_stats() or {}
        by_key: collections.Counter = collections.Counter()
        count: collections.Counter = collections.Counter()
        total = 0
        for arr in jax.live_arrays():
            if dev not in arr.devices():
                continue
            nbytes = math.prod(arr.sharding.shard_shape(arr.shape)) * arr.dtype.itemsize
            key = (tuple(arr.shape), str(arr.dtype))
            by_key[key] += nbytes
            count[key] += 1
            total += nbytes
        try:
            from jax.extend import backend as _backend

            n_exec = len(_backend.get_backend().live_executables())
        except Exception:  # noqa: BLE001
            n_exec = -1
        top = ", ".join(
            f"{list(k[0])}{k[1]} x{count[k]}={v / 2**20:.0f}M" for k, v in by_key.most_common(5)
        )
        grown = sorted(
            ((count[k] - _prev_count.get(k, 0), k) for k in count),
            key=lambda t: -t[0] * (by_key[t[1]] / max(count[t[1]], 1)),
        )
        growth = ", ".join(
            f"{list(k[0])}{k[1]} {d:+d} (x{count[k]})" for d, k in grown[:5] if d > 0
        )
        _prev_count = count
        mib = lambda key: stats.get(key, 0) / 2**20  # noqa: E731
        logger.info(
            "[memlog %s] in_use=%.0fM peak=%.0fM limit=%.0fM largest_free=%.0fM allocs=%s "
            "live_arrays=%d (%.0fM on dev0) executables=%d gc=%d top: %s | grew: %s",
            tag,
            mib("bytes_in_use"),
            mib("peak_bytes_in_use"),
            mib("bytes_limit"),
            mib("largest_free_block_bytes"),
            stats.get("num_allocs"),
            sum(count.values()),
            total / 2**20,
            n_exec,
            collected,
            top,
            growth or "-",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("memlog failed: %s", e)
